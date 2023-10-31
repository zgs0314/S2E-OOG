import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_processor import find_data_path

# adj_matrix
class GraphNet(torch.nn.Module):
    def __init__(self, num_node_features, out_channels, num_classes, args):
        super(GraphNet, self).__init__()
        self.pretrain_model = args.pretrain_model
        self.device = torch.device("cuda:"+ str(args.cuda) if torch.cuda.is_available() else "cpu")
        if self.pretrain_model == 'GCN':
            self.conv1 = GCNConv(num_node_features, 64).to(self.device)
            self.conv2 = GCNConv(64, out_channels).to(self.device)
        elif self.pretrain_model == 'GAT':
            self.conv1 = GATConv(num_node_features, 64).to(self.device)
            self.conv2 = GATConv(64, out_channels).to(self.device)
        elif self.pretrain_model == 'GraphSAGE':
            self.conv1 = GraphSAGE(num_node_features, 64, 2, out_channels).to(self.device)
        self.classifier = nn.Linear(out_channels, num_classes).to(self.device)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        if self.pretrain_model in ['GCN', 'GAT']:
            x = self.conv2(x, edge_index)
            x = F.leaky_relu(x)
        output_feat = x
        x = F.dropout(x, p=0.3, training=self.training)
        pred = self.classifier(output_feat)
        pred = F.softmax(pred, dim=1)
        return output_feat, pred
    

class PretrainFeature():
    def __init__(self, data, in_channels, out_channels, args):
        self.data = data
        self.num_classes = data.num_classes
        self.name = args.dataset
        self.path = find_data_path(args.dataset)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.total_graph_node_num = 0
        self.device = args.cuda
        self.pretrain_model = args.pretrain_model
        self.pretrain_with_feature = args.pretrain_with_feature
        
        self.edge_index_all = None
        self.num_seen = []
        self.label_seen = []
        self.edge_index_seen = None
        self.output_feat_seen = None
        
        self.model = GraphNet(self.in_channels, self.out_channels, self.num_classes, args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.007, weight_decay=5e-5)
        self.loss_fn = F.cross_entropy
    
    def edit_seen_graph(self):
        # input edges
        self.ntype = self.data.train_graph.ntypes
        self.old_map_new = {ntype:{} for ntype in self.ntype}
        self.node_record_idx = {ntype:np.array([-1]) for ntype in self.ntype}
        ### number of nodes in each class
        self.node_class_num = {ntype:0 for ntype in self.ntype}
        self.type_idx = {ntype:0 for ntype in self.ntype}
        ### collect idx of each ntype
        for type in self.data.train_graph.canonical_etypes:
            edge_idx_tmp = np.array(self.data.train_edges[type])
            self.node_record_idx[type[0]] = np.concatenate([self.node_record_idx[type[0]], edge_idx_tmp[0]])
            self.node_record_idx[type[2]] = np.concatenate([self.node_record_idx[type[2]], edge_idx_tmp[1]])
        
        ### select each unique number
        for t in self.ntype:
            self.node_record_idx[t] = np.unique(self.node_record_idx[t][1:])
            self.node_class_num[t] = self.node_record_idx[t].size
        
        ### map to homo graph idx
        num_all = 0 ### for count
        self.new_map_old = 0
        self.target_ntype = self.data.target_ntype
        for t in self.ntype:
            if isinstance(self.new_map_old,int):
                self.new_map_old = self.node_record_idx[t]
            else:
                self.new_map_old = np.concatenate([self.new_map_old, self.node_record_idx[t]])
            for j in range(num_all, num_all + self.node_class_num[t]):
                self.old_map_new[t][self.new_map_old[j]] = j
            if t == self.target_ntype:
                target_idx = list(set(range(num_all, num_all + self.node_class_num[t])))
            self.type_idx[t] = list(set(range(num_all, num_all + self.node_class_num[t])))
            num_all += self.node_class_num[t]
        
        # get seen data
        self.num_seen = self.data.split['seen']
        
        ### get from data because some node in OPPO is deleted and index from the file in not true
        with open(self.path + '/labels.pkl', 'rb') as f:
            labels_split = pickle.load(f)

        if self.name in ['ACM', 'IMDB', 'DBLP']:
            labels = np.vstack((labels_split[0], labels_split[1], labels_split[2]))
        elif self.name in ['OPPO']:
            labels = self.data.oppo_label
        
        self.label_seen = 0
        for j in labels:
            if j[0] in self.num_seen:
                if isinstance(self.label_seen, int):
                    self.label_seen = np.array([[self.old_map_new[self.target_ntype][j[0]],j[1]]])
                else:
                    self.label_seen = np.concatenate([self.label_seen ,np.array([[self.old_map_new[self.target_ntype][j[0]],j[1]]])])
        self.label_seen = self.label_seen[np.argsort(self.label_seen, axis = 0)[:,0].tolist()].astype('int64')
        self.label_seen = torch.tensor(self.label_seen)

        ### map edges
        self.edge_index_seen_new = 0
        for type in self.data.train_graph.canonical_etypes:
            edge_idx_tmp = np.array(self.data.train_edges[type])
            for i in range(edge_idx_tmp[0].size):
                edge_idx_tmp[0][i] = self.old_map_new[type[0]][edge_idx_tmp[0][i]]
                edge_idx_tmp[1][i] = self.old_map_new[type[2]][edge_idx_tmp[1][i]]
            if isinstance(self.edge_index_seen_new, int):
                self.edge_index_seen_new = edge_idx_tmp
            else:
                self.edge_index_seen_new = np.concatenate([self.edge_index_seen_new, edge_idx_tmp],axis=1)
        self.edge_index_seen_new = torch.tensor(self.edge_index_seen_new)

        self.nc_node_idx_train_new = target_idx

        node_num = num_all
        if self.pretrain_with_feature:
            self.train_random_feat = 0
            for t in self.ntype:
                if isinstance(self.train_random_feat, int):
                    self.train_random_feat = self.data.node_feat[t][self.node_record_idx[t]]
                else:
                    self.train_random_feat = np.concatenate([self.train_random_feat, self.data.node_feat[t][self.node_record_idx[t]]])
            self.train_random_feat = torch.tensor(self.train_random_feat)
        else:
            self.train_random_feat = torch.randn(size=(node_num, self.in_channels)).float()
        
        self.train_random_feat = self.train_random_feat.to(self.device)
        self.label_seen = self.label_seen.to(self.device) ### ok
        self.nc_node_idx_train_new = torch.tensor(self.nc_node_idx_train_new).to(self.device) ### ok
        self.edge_index_seen_new = self.edge_index_seen_new.to(self.device) ### ok
        
    def pretrain(self, total_epochs):
        '''train the pretrain model'''
        for epoch in range(total_epochs+1):
            output_feat, pred = self.model(self.train_random_feat, self.edge_index_seen_new)
            lbl = self.label_seen[:,1]
            pred = pred[self.nc_node_idx_train_new]
            # masked attribute for pretraining
            label_index_list = list(range(self.label_seen[:,0].shape[0]))
            mask_rate = 0.3
            sample_num = int(self.label_seen[:,0].shape[0] * (1 - mask_rate))
            label_index_sampled = random.sample(label_index_list, sample_num)
            label_index_masked = list(set(label_index_list) - set(label_index_sampled))
            
            loss = self.loss_fn(pred[label_index_sampled], lbl[label_index_sampled])
            predict_all = pred.argmax(dim=1)
            acc_sampled = accuracy_score(lbl[label_index_sampled].to('cpu'), predict_all[label_index_sampled].to('cpu'))
            macro_f1_sampled = f1_score(lbl[label_index_sampled].to('cpu'), predict_all[label_index_sampled].to('cpu'), average='macro')
            micro_f1_sampled = f1_score(lbl[label_index_sampled].to('cpu'), predict_all[label_index_sampled].to('cpu'), average='micro')   
            
            acc_masked = accuracy_score(lbl[label_index_masked].to('cpu'), predict_all[label_index_masked].to('cpu'))
            macro_f1_masked = f1_score(lbl[label_index_masked].to('cpu'), predict_all[label_index_masked].to('cpu'), average='macro')
            micro_f1_masked = f1_score(lbl[label_index_masked].to('cpu'), predict_all[label_index_masked].to('cpu'), average='micro')   
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch % 100) == 0:
                print('{}, epoch: {}, loss: {:.3f}, sample acc: {:.3f}, mask acc: {:.3f}, macro_f1_masked: {:.3f}'.format(\
                    self.name, epoch, loss, acc_sampled, acc_masked, macro_f1_masked))

        self.out_feat = {}
        name_node_num_dict = self.data.nodenum_dict
        output_feat_arr = output_feat.cpu().detach().numpy() # to numpy
        for t in self.ntype:
            self.out_feat[t] = np.random.randn(name_node_num_dict[t], self.out_channels)
            self.out_feat[t][self.new_map_old[self.type_idx[t]]] = output_feat_arr[self.type_idx[t]]
        return self.out_feat

