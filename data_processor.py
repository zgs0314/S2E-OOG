import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import pickle
import dgl
from random import sample
import json
import os
import argparse

def find_data_path(dataset):
    ### in ACM, IMDB, the data are devided into node_features.pkl, edges.pkl, labels.pkl
    if dataset == 'ACM':
        return './dataset/acm4GTN'
    elif dataset == 'IMDB':
        return './dataset/imdb4GTN'
    elif dataset == 'OPPO':
        return './dataset/oppo'
    else:
        pass

### contents adapt from openhgnn
class NodeClassDataset(Dataset):
    def __init__(self, args):
        self.name = args.dataset
        self.path = find_data_path(args.dataset)
        ### set graph
        if self.name == 'ACM':
            canonical_etypes = [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                                ('paper', 'paper-subject', 'subject'), ('subject', 'subject-paper', 'paper')]
            target_ntype = 'paper'
            meta_paths_dict = {'PAPSP': [('paper', 'paper-author', 'author'),
                                         ('author', 'author-paper', 'paper'),
                                         ('paper', 'paper-subject', 'subject'),
                                         ('subject', 'subject-paper', 'paper')],
                               'PAP': [('paper', 'paper-author', 'author'),
                                       ('author', 'author-paper', 'paper')],
                               'PSP': [('paper', 'paper-subject', 'subject'),
                                       ('subject', 'subject-paper', 'paper')]
                               }
        elif self.name == 'IMDB':
            canonical_etypes = [('movie', 'movie-director', 'director'), ('director', 'director-movie', 'movie'),
                                ('movie', 'movie-actor', 'actor'), ('actor', 'actor-movie', 'movie')]
            target_ntype = 'movie'
            meta_paths_dict = {'MAM': [('movie', 'movie-actor', 'actor'),
                                       ('actor', 'actor-movie', 'movie')],
                               'MDM': [('movie', 'movie-director', 'director'),
                                       ('director', 'director-movie', 'movie')]
                               }
        elif self.name == 'OPPO':
            canonical_etypes = [('app', 'app-permission', 'permission'), ('permission', 'permission-app', 'app'),
                                ('app', 'app-intent', 'intent'), ('intent', 'intent-app', 'app')]
            target_ntype = 'app'
            meta_paths_dict = {'APA': [('app', 'app-permission', 'permission'),
                                       ('permission', 'permission-app', 'app')],
                               'AIA': [('app', 'app-intent', 'intent'),
                                       ('intent', 'intent-app', 'app')]
                               }
        else:
            pass
        self.canonical_etypes = canonical_etypes
        self.target_ntype = target_ntype
        self.edge_type = [i[1] for i in self.canonical_etypes]
        self.meta_paths_dict = meta_paths_dict

        if self.name in ['ACM', 'IMDB']:
            with open(self.path + '/node_features.pkl', 'rb') as f:
                node_features = pickle.load(f)
        with open(self.path + '/edges.pkl', 'rb') as f:
            edges = pickle.load(f)
        with open(self.path + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        if self.name in ['ACM', 'IMDB']:
            num_nodes = edges[0].shape[0]
        elif self.name in ['OPPO']:
            num_nodes = int(edges[2][1].max()) + 1
        assert len(canonical_etypes) == len(edges)

        ntype_mask = dict()
        ### changed by yan
        self.ntype_mask = dict()
        ntype_idmap = dict()
        ntypes = set()
        data_dict = {}

        ### add the whole index of each type there
        if self.name == 'ACM':
            self.whole_index = {'paper': np.arange(0, 3025), 
                                'author': np.arange(3025, 8937),
                                'subject': np.arange(8937, 8994)}
            self.node_all = 8994
        elif self.name == 'IMDB':
            self.whole_index = {'movie': np.arange(0, 4661), 
                                'actor': np.arange(4661, 10502),
                                'director': np.arange(10502, 12772)}
            self.node_all = 12772
        elif self.name == 'OPPO':
            self.whole_index = {'app': np.arange(0, 35027), 
                                'permission': np.arange(35027, 35894),
                                'intent': np.arange(35894, 37078)}
            self.node_all = 37078

        # create dgl graph
        for etype in canonical_etypes:
            ntypes.add(etype[0])
            ntypes.add(etype[2])
        for ntype in ntypes:
            ntype_mask[ntype] = np.zeros(num_nodes, dtype=bool)
            ntype_idmap[ntype] = np.full(num_nodes, -1, dtype=int)
        for i, etype in enumerate(canonical_etypes):
            if self.name in ['ACM', 'IMDB']:
                src_nodes = edges[i].nonzero()[0]
                dst_nodes = edges[i].nonzero()[1]
            elif self.name in ['OPPO']:
                src_nodes = edges[i][0].astype(int)
                dst_nodes = edges[i][1].astype(int)
            src_ntype = etype[0]
            dst_ntype = etype[2]
            ntype_mask[src_ntype][src_nodes] = True
            ntype_mask[dst_ntype][dst_nodes] = True
        for ntype in ntypes:
            ntype_idx = ntype_mask[ntype].nonzero()[0]
            ntype_idmap[ntype][ntype_idx] = np.arange(ntype_idx.size)
        for i, etype in enumerate(canonical_etypes):
            if self.name in ['ACM', 'IMDB']:
                src_nodes = edges[i].nonzero()[0]
                dst_nodes = edges[i].nonzero()[1]
            elif self.name in ['OPPO']:
                src_nodes = edges[i][0].astype(int)
                dst_nodes = edges[i][1].astype(int)
            src_ntype = etype[0]
            dst_ntype = etype[2]
            data_dict[etype] = \
                (torch.from_numpy(ntype_idmap[src_ntype][src_nodes]).type(torch.int64),
                 torch.from_numpy(ntype_idmap[dst_ntype][dst_nodes]).type(torch.int64))
        g = dgl.heterograph(data_dict)

        self.ntype_mask = ntype_mask
        # split and label
        if self.name in ['OPPO']:
            in_all = [902, 1018, 1309, 1337, 1351, 1444, 2583, 3093, 3155, 3173, 
                      3283, 3447, 4016, 4101, 4684, 4709, 5239, 5329, 5527, 5818,
                      5826, 6630, 6741, 6791, 6954, 7235, 7247, 7767, 8422, 8890,
                      9410, 9543, 9798, 9891, 10152, 10477, 10568, 11191, 11366, 11405,
                      11439, 11667, 11995, 12209, 12844, 13088, 13510, 13928, 14547, 14750,
                      14786, 15205, 15683, 16213, 17248, 18298]
            labels = np.delete(np.array(labels), in_all, 0)
            labels[:,0] = np.array(list(set(range(35083 - len(in_all)))))
            self.oppo_label = labels
        all_label = np.full(g.num_nodes(target_ntype), -1, dtype=int)

        ### self.nodenum_dict: number of nodes in each graph
        self.nodenum_dict = {i:g.nodes(i).shape[0] for i in g.ntypes}

        if self.name in ['ACM', 'IMDB']:
            for i, split in enumerate(['train', 'val', 'test']):
                if isinstance(labels[i],list):
                    node = np.array(labels[i])[:, 0]
                    label = np.array(labels[i])[:, 1]
                else:
                    node = labels[i][:, 0].astype(int)
                    label = labels[i][:, 1]
                all_label[node] = label
        else:
            node = np.array(labels)[:, 0].astype(int)
            label = np.array(labels)[:, 1]
            all_label[node] = label
        g.nodes[target_ntype].data['label'] = torch.from_numpy(all_label).type(torch.long)

        ### assume train and validation size
        if os.path.exists(self.path +'/cunchu.json'):
            f = open(self.path +'/cunchu.json', 'r')
            content = f.read()
            content2 = json.loads(content)
            f.close()
            num_seen = content2['seen']
            if self.name == 'IMDB':
                num_without_label = content2['seen_wolbl']
            num_train = content2['train']
            num_valid = content2['valid']
            num_test = content2['test']
            if self.name == 'IMDB':
                data_cun = {'seen':num_seen, 'seen_wolbl': num_without_label, 'train':num_train, 'valid': num_valid, 'test': num_test}
            else:
                data_cun = {'seen':num_seen, 'train': num_train, 'valid': num_valid, 'test': num_test}
        else:
            ### set out/in/graph node number
            if self.name == 'ACM':
                seen, train_n, valid_n = 1500, 300, 300
            elif self.name == 'IMDB':
                seen, train_n, valid_n = 1478, 300, 300
            elif self.name == 'OPPO':
                seen, train_n, valid_n = 28000, 1400, 1400
            else:
                pass
            num_predict = self.nodenum_dict[target_ntype]
            if self.name == 'IMDB':
                num_all_over = list(set(range(num_predict)))
                num_all = torch.nonzero(g.nodes[target_ntype].data['label']+1).squeeze(1).tolist()
                num_without_label = list(set(num_all_over) - set(num_all))
            else:
                num_all = list(set(range(num_predict)))
            num_seen_train_valid = sample(num_all, seen + train_n + valid_n)
            num_seen = sample(num_seen_train_valid, seen)
            num_train_valid = list(set(num_seen_train_valid) - set(num_seen))
            num_train = sample(num_train_valid, train_n)
            num_valid = list(set(num_train_valid) - set(num_train))
            num_test = list(set(num_all) - set(num_seen_train_valid))
            if self.name == 'IMDB':
                data_cun = {'seen':num_seen, 'seen_wolbl': num_without_label, 'train':num_train, 'valid': num_valid, 'test': num_test}
            else:
                data_cun = {'seen':num_seen, 'train':num_train, 'valid': num_valid, 'test': num_test}
            data_cun_0 = json.dumps(data_cun)
            f2 = open(self.path +'/cunchu.json', 'w')
            f2.write(data_cun_0)
            f2.close()
        self.split = data_cun

        self.node_feat = {}
        # node feature
        if args.with_feature:
            node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
            for ntype in ntypes:
                idx = ntype_mask[ntype].nonzero()[0]
                g.nodes[ntype].data['h'] = node_features[idx]
                ### for common process
                self.node_feat[ntype] = node_features[idx]
        else:
            ### random initialized feature
            if args.learnable_feature:
                ### baseline: neighbor as feature and linear
                pass
            else:
                use_random_feat = True
                if use_random_feat:
                    for ntype in ntypes:
                    ### when without feature, randomly initialize the feature
                        print(ntype + '_random')
                        dim_feat = 128
                        num_feat = len(ntype_mask[ntype].nonzero()[0])
                        g.nodes[ntype].data['h'] = torch.rand([num_feat, dim_feat])
                        self.node_feat[ntype] = g.nodes[ntype].data['h']
                else:
                    dname = args.dataset.lower()
                    pre_feat = np.load(self.path + f'/init_feat_seen_{dname}.npz')['arr_0']
                    for ntype in ntypes:
                        idx = ntype_mask[ntype].nonzero()[0]
                        g.nodes[ntype].data['h'] = torch.tensor(pre_feat[idx],dtype=torch.float32)
                        self.node_feat[ntype] = torch.tensor(pre_feat[idx],dtype=torch.float32)

        ### preprocess for raw feature
        if args.raw_feat:
            node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
            for ntype in ntypes:
                if ntype == self.target_ntype:
                    idx = ntype_mask[ntype].nonzero()[0]
                    # g.nodes[ntype].data['h'] = node_features[idx]
                    ### for common process
                    self.raw_feat = node_features[idx]

        self.g = g # whole graph
        self.num_classes = len(torch.unique(self.g.nodes[self.target_ntype].data['label']))
        if self.name == 'IMDB': ### some nodes have no label
            self.num_classes = self.num_classes - 1
        if not args.with_feature and args.learnable_feature:
            self.in_dim = 128
        else:
            self.in_dim = self.g.ndata['h'][self.target_ntype].shape[1]
        

        ### get node: g.nodes('author')
        ### get edge: g.edges(etype = 'author-paper')
        ### add some message for save here, add a neighbor dict (for each target node, their neighbors are ...)
        self.neighbor_dict = {}
        if self.name == 'OPPO' and os.path.exists(self.path + '/neighbor_dict.json'):
            f = open(self.path + '/neighbor_dict.json', 'r')
            content = f.read()
            self.neighbor_dict = json.loads(content)
            for j in list(self.neighbor_dict):
                self.neighbor_dict[int(j)] = self.neighbor_dict.pop(j)
            f.close()
        else:
            for j in data_dict:
                if j[2] == self.target_ntype:
                    for k in range(len(data_dict[j][0])):
                        target_node_0 = data_dict[j][1][k].item()
                        if target_node_0 not in self.neighbor_dict:
                            self.neighbor_dict[target_node_0] = {}
                        if j[0] not in self.neighbor_dict[target_node_0]:
                            self.neighbor_dict[target_node_0][j[0]] = []
                        self.neighbor_dict[target_node_0][j[0]].append(data_dict[j][0][k].item())
                else:
                    pass
            if self.name == 'OPPO':
                f3 = open(self.path +'/neighbor_dict.json', 'w')
                neighbor_dict = json.dumps(self.neighbor_dict)
                f3.write(neighbor_dict)
                f3.close()

        ### process the train graph || preprocess the valid/test nodes
        ### self.train_graph: the DGL graph for training
        ### self.train_edges: the dict that contains the edges in the train graph(maybe later used for pyg)
        ### self.train/valid/test_node_record: record the edges that connects target unseen train/valid/test node
        # if args.inductive:
        self.train_edges = {}
        self.all_edges = {}
        for type in self.g.canonical_etypes:
            self.all_edges[type] = [self.g.edges(etype = type[1])[0].tolist(), self.g.edges(etype = type[1])[1].tolist()]
        if 1:
            ### process for each dataset
            if self.name == 'ACM':
                train_g_init = dgl.sampling.sample_neighbors(g,{'paper':num_seen}, -1)
                ### the edges in the train graph
                edges_author_paper = train_g_init.edges(etype = 'author-paper')
                edges_subject_paper = train_g_init.edges(etype = 'subject-paper')
                train_g_init = dgl.add_edges(train_g_init, edges_author_paper[1], edges_author_paper[0], etype = 'paper-author')
                train_g_init = dgl.add_edges(train_g_init, edges_subject_paper[1], edges_subject_paper[0], etype = 'paper-subject')
                self.train_graph = train_g_init
                ### in the inductive setting, it is not proper to test the train data(transductive things)
                valid_test_record = [{},{},{}]
                for type in self.train_graph.canonical_etypes:
                    if self.target_ntype in type:
                        edge_idx = [self.g.edges(etype = type[1])[0].tolist(), self.g.edges(etype = type[1])[1].tolist()]
                        self.train_edges[type] = [self.train_graph.edges(etype = type[1])[0].tolist(), self.train_graph.edges(etype = type[1])[1].tolist()]
                        if self.target_ntype == type[0]: ### place of the target type
                            show_idx = 0
                        else:
                            show_idx = 1
                        ### get the target type indexes, and put the corresponding node to the 
                        for j in range(len(edge_idx[show_idx])):
                            i = edge_idx[show_idx][j]
                            if i in num_valid:
                                dict_idx = 0
                            elif i in num_test:
                                dict_idx = 1
                            elif i in num_train:
                                dict_idx = 2
                            else:
                                dict_idx = 3
                            if dict_idx in [0,1,2]:
                            ### the node not exist, create one
                                if i not in valid_test_record[dict_idx]:
                                    valid_test_record[dict_idx][i] = {}
                                if type[1] not in valid_test_record[dict_idx][i]:
                                    valid_test_record[dict_idx][i][type[1]] = [[],[]]
                                valid_test_record[dict_idx][i][type[1]][show_idx].append(i)
                                valid_test_record[dict_idx][i][type[1]][1-show_idx].append(edge_idx[1-show_idx][j])
                self.valid_node_record = valid_test_record[0]
                self.test_node_record = valid_test_record[1]
                self.train_node_record = valid_test_record[2]
            elif self.name == 'IMDB':
                train_g_init = dgl.sampling.sample_neighbors(g,{'movie':num_seen + num_without_label}, -1)
                edges_actor_movie = train_g_init.edges(etype = 'actor-movie')
                edges_director_movie = train_g_init.edges(etype = 'director-movie')
                train_g_init = dgl.add_edges(train_g_init, edges_actor_movie[1], edges_actor_movie[0], etype = 'movie-actor')
                train_g_init = dgl.add_edges(train_g_init, edges_director_movie[1], edges_director_movie[0], etype = 'movie-director')
                self.train_graph = train_g_init
                valid_test_record = [{},{},{}]
                for type in self.train_graph.canonical_etypes:
                    if self.target_ntype in type:
                        edge_idx = [self.g.edges(etype = type[1])[0].tolist(), self.g.edges(etype = type[1])[1].tolist()]
                        self.train_edges[type] = [self.train_graph.edges(etype = type[1])[0].tolist(), self.train_graph.edges(etype = type[1])[1].tolist()]
                        if self.target_ntype == type[0]: ### place of the target type
                            show_idx = 0
                        else:
                            show_idx = 1
                        ### get the target type indexes, and put the corresponding node to the 
                        for j in range(len(edge_idx[show_idx])):
                            i = edge_idx[show_idx][j]
                            if i in num_valid:
                                dict_idx = 0
                            elif i in num_test:
                                dict_idx = 1
                            elif i in num_train:
                                dict_idx = 2
                            else:
                                dict_idx = 3
                            if dict_idx in [0,1,2]:
                            ### the node not exist, create one
                                if i not in valid_test_record[dict_idx]:
                                    valid_test_record[dict_idx][i] = {}
                                if type[1] not in valid_test_record[dict_idx][i]:
                                    valid_test_record[dict_idx][i][type[1]] = [[],[]]
                                valid_test_record[dict_idx][i][type[1]][show_idx].append(i)
                                valid_test_record[dict_idx][i][type[1]][1-show_idx].append(edge_idx[1-show_idx][j])
                self.valid_node_record = valid_test_record[0]
                self.test_node_record = valid_test_record[1] 
                self.train_node_record = valid_test_record[2]           
                pass
            elif self.name == 'OPPO':
                train_g_init = dgl.sampling.sample_neighbors(g,{'app':num_seen}, -1)
                edges_permission_app = train_g_init.edges(etype = 'permission-app')
                edges_intent_app = train_g_init.edges(etype = 'intent-app')
                train_g_init = dgl.add_edges(train_g_init, edges_permission_app[1], edges_permission_app[0], etype = 'app-permission')
                train_g_init = dgl.add_edges(train_g_init, edges_intent_app[1], edges_intent_app[0], etype = 'app-intent')
                self.train_graph = train_g_init
                for type in self.train_graph.canonical_etypes:
                    edge_idx = [self.train_graph.edges(etype = type[1])[0].tolist(), self.train_graph.edges(etype = type[1])[1].tolist()]
                    self.train_edges[type] = edge_idx
                if os.path.exists(self.path + '/valid_test_cun.json'):
                    f = open(self.path + '/valid_test_cun.json', 'r')
                    content = f.read()
                    valid_test_record = json.loads(content)
                    for j in valid_test_record:
                        for k in list(j):
                            j[int(k)] = j.pop(k)
                    f.close()
                else:
                    ### in the inductive setting, it is not proper to test the train data(transductive things)
                    valid_test_record = [{},{},{}]
                    for type in self.train_graph.canonical_etypes:
                        if self.target_ntype in type:
                            edge_idx = [self.g.edges(etype = type[1])[0].tolist(), self.g.edges(etype = type[1])[1].tolist()]
                            if self.target_ntype == type[0]: ### place of the target type
                                show_idx = 0
                            else:
                                show_idx = 1
                            ### get the target type indexes, and put the corresponding node to the 
                            for j in range(len(edge_idx[show_idx])):
                                i = edge_idx[show_idx][j]
                                if i in num_valid:
                                    dict_idx = 0
                                elif i in num_test:
                                    dict_idx = 1
                                elif i in num_train:
                                    dict_idx = 2
                                else:
                                    dict_idx = 3
                                if dict_idx in [0,1,2]:
                                ### the node not exist, create one
                                    if i not in valid_test_record[dict_idx]:
                                        valid_test_record[dict_idx][i] = {}
                                    if type[1] not in valid_test_record[dict_idx][i]:
                                        valid_test_record[dict_idx][i][type[1]] = [[],[]]
                                    valid_test_record[dict_idx][i][type[1]][show_idx].append(i)
                                    valid_test_record[dict_idx][i][type[1]][1-show_idx].append(edge_idx[1-show_idx][j])
                    f3 = open(self.path +'/valid_test_cun.json', 'w')
                    valid_test_record_0 = json.dumps(valid_test_record)
                    f3.write(valid_test_record_0)
                    f3.close()
                self.valid_node_record = valid_test_record[0]
                self.test_node_record = valid_test_record[1]
                self.train_node_record = valid_test_record[2]    

        aaaa = 0

    ### return split
    def get_split(self):
        return torch.tensor(self.split['train']), torch.tensor(self.split['valid']), torch.tensor(self.split['test'])

    ### return labels
    def get_labels(self):
        return self.g.nodes[self.target_ntype].data['label']

    def get_node_class_num(self):
        n_dict = {}
        for n in self.train_graph.ntypes:
            n_dict[n] = self.train_graph.num_nodes(n)
        return n_dict
