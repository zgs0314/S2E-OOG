import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GraphSAGE, GATConv, GCNConv

class SAGE_model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SAGE_model, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gnn = GraphSAGE(self.in_dim, self.hidden_dim, 1, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        pass

    def forward(self, embedding, edges):
        emb0 = self.gnn(embedding, edges)
        logits = self.classifier(emb0)[0].unsqueeze(0)
        return logits

class GAT_model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GAT_model, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gnn = GATConv(in_dim, hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        pass

    def forward(self, embedding, edges):
        emb0 = self.gnn(embedding, edges)
        logits = self.classifier(emb0)[0].unsqueeze(0)
        return logits
    
    
class GCN_model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN_model, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gcn = GCNConv(in_dim, hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        pass

    def forward(self, embedding, edges):
        emb0 = self.gcn(embedding, edges)
        logits = self.classifier(emb0)[0].unsqueeze(0)
        return logits

class SingleModel(nn.Module): ### the model of mix
    def __init__(self, device, init_dim, hidden_dim, out_dim, arch_dict, ntype_list=[]):
        super(SingleModel, self).__init__()
        ### whether uniform layer
        self.device = device
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.ntype_list = ntype_list
        self.arch_dict = arch_dict
        self.lenn = len(arch_dict['trans'])
        # transition function: id, linear, rel-linear, rel-linear-conv
        self.trans1 = arch_dict['trans'][0]
        self.trans2 = arch_dict['trans'][1]
        if self.lenn == 3:
            self.trans3 = arch_dict['trans'][2]
        # activate function: relu, tanh, sigmoid
        self.activate_func = arch_dict['activate']
        # aggregate node: SAGE-mean, SAGE-sum, SAGE-max, GAT, GIN
        self.agg_node = arch_dict['agg_node']
        self.construct_trans_module()
        # node classification task 
        self.classifier = nn.Linear(hidden_dim, out_dim)
        
    def construct_trans_module(self):
        if self.trans1 == 'linear':
            layer_linear = nn.Linear(self.init_dim, self.hidden_dim)
            self.add_module('linear1', layer_linear)
        elif self.trans1 == 'conv':
            layer_conv = nn.Conv1d(self.init_dim, self.hidden_dim, self.ker_dim, padding=1)
            self.add_module('conv1'+i, layer_conv)
        elif self.trans1 in ['rel-linear', 'rel-linear-conv']:
            for i in self.ntype_list:
                if self.trans1 == 'rel-linear-conv':
                    self.ker_dim = 3
                    # conv layer for init
                    conv_add = nn.Conv1d(self.init_dim, self.init_dim, self.ker_dim, padding = 1)
                    self.add_module('conv1' + i, conv_add)
                layer_trans = nn.Linear(self.init_dim, self.hidden_dim)# trans linear layer
                self.add_module('linear1' + i, layer_trans)
        else:
            pass

        if self.trans2 == 'linear':
            layer_linear = nn.Linear(self.init_dim, self.hidden_dim)
            self.add_module('linear2', layer_linear)
        elif self.trans2 == 'conv':
            layer_conv = nn.Conv1d(self.init_dim, self.hidden_dim, self.ker_dim, padding=1)
            self.add_module('conv2'+i, layer_conv)
        elif self.trans2 in ['rel-linear', 'rel-linear-conv']:
            for i in self.ntype_list:
                if self.trans2 == 'rel-linear-conv':
                    self.ker_dim = 3
                    # conv layer for init
                    conv_add = nn.Conv1d(self.init_dim, self.init_dim, self.ker_dim, padding = 1)
                    self.add_module('conv2' + i, conv_add)
                layer_trans = nn.Linear(self.init_dim, self.hidden_dim)# trans linear layer
                self.add_module('linear2' + i, layer_trans)
        else:
            pass

        if self.lenn == 3:
            if self.trans3 == 'linear':
                layer_linear = nn.Linear(self.init_dim, self.hidden_dim)
                self.add_module('linear3', layer_linear)
            elif self.trans3 == 'conv':
                layer_conv = nn.Conv1d(self.init_dim, self.hidden_dim, self.ker_dim, padding=1)
                self.add_module('conv3'+i, layer_conv)
            elif self.trans3 in ['rel-linear', 'rel-linear-conv']:
                for i in self.ntype_list:
                    if self.trans3 == 'rel-linear-conv':
                        self.ker_dim = 3
                        # conv layer for init
                        conv_add = nn.Conv1d(self.init_dim, self.init_dim, self.ker_dim, padding = 1)
                        self.add_module('conv3' + i, conv_add)
                    layer_trans = nn.Linear(self.init_dim, self.hidden_dim)# trans linear layer
                    self.add_module('linear3' + i, layer_trans)
            else:
                pass

    def forward(self, feat, neighbor_dict, all_feat):
        logits = 0
        feat_agg = {}
        feat_input = {j:all_feat[j][neighbor_dict[j]] for j in neighbor_dict}
        feat_out_1 = {}
        feat_out_2 = {}
        if self.trans1 == 'linear':
            for j in feat_input:
                feat_agg[j] = self._modules['linear1'](feat_input[j])
        elif self.trans1 == 'conv':
            for j in feat_input:
                feat_agg[j] = self._modules['conv1'](feat_input[j].transpose(0,1)).transpose(0,1)
        elif self.trans1 in ['rel-linear']:
            for j in feat_input:
                feat_agg[j] = self._modules['linear1'+j](feat_input[j])          
        elif self.trans1 in ['rel-conv']:
            for j in feat_input:
                feat_agg[j] = self._modules['conv1'+j](feat_input[j].transpose(0,1)).transpose(0,1)
        else:
            feat_agg = feat_input

        if self.trans2 == 'linear':
            for j in feat_agg:
                feat_out_1[j] = self._modules['linear2'](feat_agg[j])
        elif self.trans2 == 'conv':
            for j in feat_agg:
                feat_out_1[j] = self._modules['conv2'](feat_agg[j].transpose(0,1)).transpose(0,1)
        elif self.trans2 in ['rel-linear']:
            for j in feat_agg:
                feat_out_1[j] = self._modules['linear2'+j](feat_agg[j])          
        elif self.trans2 in ['rel-conv']:
            for j in feat_agg:
                feat_out_1[j] = self._modules['conv2'+j](feat_agg[j].transpose(0,1)).transpose(0,1)
        else:
            feat_out_1 = feat_agg

        if self.lenn == 3:
            if self.trans3 == 'linear':
                for j in feat_agg:
                    feat_out_2[j] = self._modules['linear3'](feat_out_1[j])
            elif self.trans3 == 'conv':
                for j in feat_agg:
                    feat_out_2[j] = self._modules['conv3'](feat_out_1[j].transpose(0,1)).transpose(0,1)
            elif self.trans3 in ['rel-linear']:
                for j in feat_agg:
                    feat_out_2[j] = self._modules['linear3'+j](feat_out_1[j])          
            elif self.trans2 in ['rel-conv']:
                for j in feat_agg:
                    feat_out_2[j] = self._modules['conv3'+j](feat_out_1[j].transpose(0,1)).transpose(0,1)
            else:
                feat_out_2 = feat_out_1

        if self.lenn == 3:
            feat_agg_0 = torch.cat([feat_out_2[j] for j in feat_out_2])
        else:
            feat_agg_0 = torch.cat([feat_out_1[j] for j in feat_out_1])

        if self.activate_func == 'relu':
            feat_agg_0 = F.relu(feat_agg_0)
        elif self.activate_func == 'leakyrelu':
            feat_agg_0 = F.leaky_relu(feat_agg_0, negative_slope=0.1) 
        elif self.activate_func == 'tanh':
            feat_agg_0 = torch.tanh(feat_agg_0)
        elif self.activate_func == 'sigmoid':
            feat_agg_0 = torch.sigmoid(feat_agg_0)

        
        if self.agg_node == 'mean':
            feat_agg_0 = feat_agg_0.mean(dim = 0)
        elif self.agg_node == 'max':
            feat_agg_0, _ = feat_agg_0.max(dim = 0)
        elif self.agg_node == 'sum':
            feat_agg_0 = feat_agg_0.sum(dim = 0)
        elif self.agg_node == 'min':
            feat_agg_0 = feat_agg_0.min(dim = 0)

        feat_all = F.relu(feat_agg_0) 
        logits = self.classifier(feat_all)
        return logits

