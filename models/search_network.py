import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from collections import OrderedDict
import itertools
from torch_geometric.nn import GraphSAGE, GATConv
from torch.autograd import Variable

TRANS_OPTION = [
    'linear', 'rel-linear', 'conv', 'rel-conv', 'identity'#
] # for seen nodes

AGG_OPTION = [
    'sum', 'mean', 'max'#
]

TRANS_OPTION_2 = [
    'linear', 'conv', 'gate'#
] # for unseen node feature

TRANS_OPTION_SUB = [
    'rel-linear'#
] # for seen nodes

AGG_OPTION_SUB = [
    'mean'
]

TRANS_OPTION_2_SUB = [
    'linear'
] # for unseen node feature


class TransOp(nn.Module):
    '''all trans for pretrain embedding'''
    def __init__(self, in_dim, out_dim, use_rel, ntype_list, transop_ablation): 
        super(TransOp, self).__init__()
        self._ops = nn.ModuleList()
        self.transop_ablation = transop_ablation
        
        if self.transop_ablation:
            self.trans_dict = TRANS_OPTION_SUB
        else:
            self.trans_dict = TRANS_OPTION
        if use_rel:#always True
            for trans in self.trans_dict:
                if trans == 'linear':
                    op = Rel_Trans(in_dim, out_dim, ntype_list, rel = False)
                    self._ops.append(op)
                elif trans in ['rel-linear', 'rel-linear-conv']:
                    if trans == 'rel-linear':
                        op = Rel_Trans(in_dim, out_dim, ntype_list, rel = True)
                        self._ops.append(op)
                    else: 
                        op = Rel_Trans(in_dim, out_dim, ntype_list, rel=True, conv=True)
                        self._ops.append(op)
                elif trans in ['conv', 'rel-conv']:
                    if trans == 'rel-conv':
                        op = Conv_Trans(in_dim, out_dim, ntype_list, rel=True)
                        self._ops.append(op)
                    else:
                        op = Conv_Trans(in_dim, out_dim, ntype_list, rel=False)
                        self._ops.append(op)
                elif trans in ['gate']:
                    op = Gate_Trans(in_dim, out_dim, ntype_list, rel=True)
                    self._ops.append(op)
                else: ### for identity
                    op = Identity()
                    self._ops.append(op)
        else: ### selection for raw feature
            pass
    
    def forward(self, data, weights):
        mixed = []
        for op in self._ops:
            mixed.append(op(data))
        return self.class_sum(mixed, weights)

    def class_sum(self, list_init, weight):
        dict_0 = {}
        flag = 0
        n = 0
        for i in list_init:
            if flag:
                for j in i:
                    dict_0[j] = dict_0[j] + weight[n] * i[j]
            else:
                for j in i:
                    dict_0[j] = weight[n] * i[j]
                flag = 1
            n = n + 1
        return dict_0

class Rel_Trans(nn.Module):
    def __init__(self, in_dim, out_dim, ntype_list, rel=True, conv=False):
        super(Rel_Trans, self).__init__()
        self.conv = conv
        self.rel = rel
        if self.rel:
            for i in ntype_list:
                layer_trans = nn.Linear(in_dim, out_dim)
                self.add_module('linear' + i, layer_trans)
                if self.conv:
                    self.ker_dim = 3
                    conv_add = nn.Conv1d(in_dim, in_dim, self.ker_dim, padding = 1)
                    self.add_module('conv' + i, conv_add)
        else:
            self.lin = nn.Linear(in_dim, out_dim)
    
    def forward(self, data):
        feat_agg = {}
        if self.rel:
            if self.conv:
                for j in data:
                    feat_out = self._modules['conv'+j](data[j].transpose(0,1)).transpose(0,1) # 矩阵转置
                    feat_agg[j] = F.relu(self._modules['linear'+j](feat_out))
            else:
                for j in data: # use feature 
                    feat_agg[j] = F.relu(self._modules['linear'+j](data[j]))
        else:
            for j in data:
                feat_agg[j] = F.relu(self.lin(data[j]))
        return feat_agg


class Conv_Trans(nn.Module):
    def __init__(self, in_dim, out_dim, ntype_list, rel=False):
        super(Conv_Trans, self).__init__()
        self.rel = rel
        self.ker_dim = 3
        if self.rel:
            for i in ntype_list:
                conv_add = nn.Conv1d(in_dim, out_dim, self.ker_dim, padding=1)
                self.add_module('conv'+i, conv_add)
        else:
            self.conv_layer = nn.Conv1d(in_dim, out_dim, self.ker_dim, padding=1)
    
    def forward(self, data):
        feat_agg = {}
        if self.rel:#use relation
            for j in data:
                feat_out = self._modules['conv'+j](data[j].transpose(0,1)).transpose(0,1)
                feat_agg[j] = F.relu(feat_out)
        else:
            for j in data:
                feat_out = self.conv_layer(data[j].transpose(0,1)).transpose(0,1)
                feat_agg[j] = F.relu(feat_out)
        return feat_agg



class Gate_Trans(nn.Module):
    def __init__(self, in_dim, out_dim, ntype_list, rel=True):
        super(Gate_Trans, self).__init__()
        self.rel = rel
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
    
    def forward(self, data):
        feat_agg = {}
        for j in data:
            x = F.relu(self.linear1(data[j]))
            x = x * data[j]
            feat_agg[j] = self.linear2(x)
        return feat_agg


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Agg_Op(nn.Module): ### need to be specified
    def __init__(self, aggop_ablation):
        super(Agg_Op, self).__init__()
        self._ops = nn.ModuleList()
        self.aggop_ablation = aggop_ablation
        if self.aggop_ablation:
            self.agg_dict = AGG_OPTION_SUB
        else:
            self.agg_dict = AGG_OPTION

    def forward(self, data, weights):
        mixed = []
        flag = 0
        for j in data:
            if flag:
                all_cat = torch.cat([all_cat, data[j]], dim = 0)
            else:
                all_cat = data[j]
                flag = 1
        for w, op in zip(weights, self.agg_dict):
            if op == 'sum':
                x_out = w * all_cat.sum(dim = 0)
            elif op == 'max':
                xx, _ = all_cat.max(dim = 0)
                x_out = w * xx
            elif op == 'mean':
                x_out = w * all_cat.mean(dim = 0)
            elif op == 'attention':
                q_matrix, k_matrix, v_matrix = all_cat, all_cat, all_cat
                dim_k = k_matrix.shape[1]
                att = F.softmax(q_matrix @ k_matrix.T/dim_k, dim=1) @ v_matrix
                x_out = w * att.mean(dim = 0)
            else:
                pass
            mixed.append(x_out)
        return sum(mixed)

class Conv_Trans_Raw(nn.Module):
    def __init__(self, in_dim, out_dim, ker_dim, padding):
        super(Conv_Trans_Raw, self).__init__()
        self.ker_dim = ker_dim
        self.padding = padding
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv_layer = nn.Conv1d(in_dim, out_dim, self.ker_dim, padding=1)
    
    def forward(self, data):
        feat_out = self.conv_layer(data.reshape(-1,1))
        feat_agg = F.relu(feat_out).reshape(self.out_dim)
        return feat_agg

class Gate_Trans_Raw(nn.Module):
    def __init__(self, in_dim, out_dim, rel=True):
        super(Gate_Trans_Raw, self).__init__()
        self.rel = rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
    
    def forward(self, data):
        x = F.relu(self.linear1(data.reshape(1,-1)))
        x = x * data
        feat_agg = self.linear2(x).reshape(self.out_dim)
        return feat_agg

class Trans_op_2(nn.Module):
    '''trans for raw feature'''
    def __init__(self, in_dim, out_dim,transop2_ablation):
        super(Trans_op_2, self).__init__()
        self.transop2_ablation = transop2_ablation
        if self.transop2_ablation:
            self.trans_dict = TRANS_OPTION_2_SUB
        else:
            self.trans_dict = TRANS_OPTION_2
        self._ops = nn.ModuleList()
        for trans in self.trans_dict:
            if trans == 'identity':
                self._ops.append(Identity())
            elif trans == 'linear':
                self._ops.append(nn.Linear(in_dim, out_dim))
            elif trans == 'conv':
                self.ker_dim = 3
                self._ops.append(Conv_Trans_Raw(in_dim, out_dim, self.ker_dim, padding=1))
            elif trans == 'gate':
                self._ops.append(Gate_Trans_Raw(in_dim, out_dim, rel=True))
            else:
                pass
            pass

    def forward(self, data, weights):
        res = []
        for w, op in zip(weights, self._ops):
            # print('op: {}'.format(op))
            res.append(w * F.relu(op(data)))
        return sum(res)

### need to preprocess the data
class Search_Network(nn.Module): ### the model of mix
    def __init__(self, device, init_dim, hidden_dim, out_dim, args, ntype_list=[]):
        super(Search_Network, self).__init__()
        self.device = device
        self.init_dim = init_dim 
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        self.ntype_list = ntype_list

        self.raw_feat = args.raw_feat
        self.rf_num = args.rf_trans_num

        ### add a term
        self.trans_num = args.trans_num

        ## temperature
        self.temp = args.search_temp
        
        #pre-process Node 0
        self.lin1 = nn.Linear(init_dim, hidden_dim)
        
        ### use ablation
        self.transop_ablation = args.transop_ablation
        self.aggop_ablation = args.aggop_ablation
        self.transop2_ablation = args.transop2_ablation

        #transition op
        self.trans_op = nn.ModuleList()
        for i in range(self.trans_num):
            self.trans_op.append(TransOp(hidden_dim, hidden_dim, True, ntype_list, self.transop_ablation))
        
        #agg op (one layer) could add more choices
        self.agg_op = Agg_Op(self.aggop_ablation)

        #raw feature process
        if self.raw_feat:
            self.trans_op_0 = nn.ModuleList()
            self.pre_linear = nn.Linear(args.raw_dim, hidden_dim)
            for i in range(self.rf_num):
                self.trans_op_0.append(Trans_op_2(hidden_dim, hidden_dim, self.transop2_ablation))

        self.classifier = nn.Linear(hidden_dim, out_dim)

        self._initialize_alphas()
    
    def forward(self, data):
        self.tr_weights = self._get_softmax_temp(self.tr_alphas)
        self.agg_weights = self._get_softmax_temp(self.agg_alphas)
        if self.raw_feat:
            self.tr_weights_2 = self._get_softmax_temp(self.tr_alphas_2)
            data_0, r_feat = data
        else:
            data_0 = data
        output = 0
        for i in range(self.trans_num):
            data_0 = self.trans_op[i](data_0, self.tr_weights[i])

        agg_out = self.agg_op(data_0, self.agg_weights[0])

        if self.raw_feat:
            rf_out = self.pre_linear(r_feat)
            for i in range(self.rf_num):
                rf_out = self.trans_op_0[i](rf_out, self.tr_weights_2[i])
            agg_out = agg_out + rf_out

        output = self.classifier(agg_out)

        return output

    def arch_parameters(self):
        return self._arch_parameters

    def _get_softmax_temp(self, alpha):
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax(alpha / self.temp) # temperature
        # one_hot = softmax(alpha / 0.001)
        return one_hot

    ### for architecture parameter
    def _initialize_alphas(self):
        if self.transop_ablation:
            num_tr_ops = len(TRANS_OPTION_SUB)
        else:
            num_tr_ops = len(TRANS_OPTION)
        
        if self.aggop_ablation:
            num_agg_ops = len(AGG_OPTION_SUB)
        else:
            num_agg_ops = len(AGG_OPTION)
        
        if self.raw_feat:
            if self.transop2_ablation:
                num_tr_ops_2 = len(TRANS_OPTION_2_SUB)
            else:
                num_tr_ops_2 = len(TRANS_OPTION_2)

        self.tr_alphas = Variable(1e-3 * torch.randn(self.trans_num, num_tr_ops).to(self.device), requires_grad=True)
        self.agg_alphas = Variable(1e-3 * torch.randn(1, num_agg_ops).to(self.device), requires_grad=True)

        if self.raw_feat:
            self.tr_alphas_2 = Variable(1e-3 * torch.randn(self.rf_num, num_tr_ops_2).to(self.device), requires_grad=True)
            self._arch_parameters = [self.tr_alphas, self.agg_alphas, self.tr_alphas_2]
        else:
            self._arch_parameters = [self.tr_alphas, self.agg_alphas]

    def sparse_single(self, weights, opsets):
        gene = []
        indices = torch.argmax(weights, dim=-1)
        for k in indices:
            gene.append(opsets[k])
        return gene

    ### get model structure
    def genotype(self, sample=False):
        gene = []
        if self.transop_ablation:
            gene += self.sparse_single(F.softmax(self.tr_alphas, dim=-1).data.cpu(), TRANS_OPTION_SUB)
        else:
            gene += self.sparse_single(F.softmax(self.tr_alphas, dim=-1).data.cpu(), TRANS_OPTION)
        
        if self.aggop_ablation:
            gene += self.sparse_single(F.softmax(self.agg_alphas, dim=-1).data.cpu(), AGG_OPTION_SUB)
        else:
            gene += self.sparse_single(F.softmax(self.agg_alphas, dim=-1).data.cpu(), AGG_OPTION)
        
        if self.raw_feat:
            if self.transop2_ablation:
                gene += self.sparse_single(F.softmax(self.tr_alphas_2, dim=-1).data.cpu(), TRANS_OPTION_2_SUB)
            else:
                gene += self.sparse_single(F.softmax(self.tr_alphas_2, dim=-1).data.cpu(), TRANS_OPTION_2)

        return '||'.join(gene)




