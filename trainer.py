import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_processor import NodeClassDataset
from models import SAGE_model, GAT_model, Search_Network, RGCN, HeteroFeature
from models.rela_model import SingleModel, GCN_model # for search
import dgl
import os
from sklearn.metrics import f1_score, accuracy_score
from pretrainfeature import PretrainFeature
import logging
import time
import utils
from utils import init_logger
import torch.backends.cudnn as cudnn


class NC_Trainer(nn.Module):
    def __init__(self, args):
        super(NC_Trainer, self).__init__()
        self.args = args
        self.method = self.args.method
        self.pretrain_model = self.args.pretrain_model
        self.device = torch.device("cuda:"+ str(args.cuda) if torch.cuda.is_available() else "cpu")
        
        self.data = NodeClassDataset(args)
        self.init_feat = {i: self.data.node_feat[i].to(self.device) for i in self.data.node_feat}

        if self.pretrain_model in ['GCN', 'GAT', 'GraphSAGE']:
            in_channels = self.data.in_dim
            self.pre_train = PretrainFeature(self.data, in_channels=in_channels, out_channels = self.args.in_dim, args=args)
        else: 
            self.train_model = build_model(args, self.data, self.device, self.pretrain_model).to(self.device)
            self.train_optimizer = optim.AdamW(self.train_model.parameters(), lr=args.het_prelr, weight_decay=args.het_prewd)
            self.train_loss_fn = F.cross_entropy 
        
        ### the data class, to get the information, only need 
        self.target = self.data.target_ntype
        self.train_idx, self.valid_idx, self.test_idx = self.data.get_split()
        self.labels = self.data.get_labels()

        self.inductive = args.inductive
        self.reg_gamma = args.reg_gamma
        if self.inductive:
            self.train_graph = self.data.train_graph
        else:
            self.train_graph = self.data.g

        if self.method not in ['Base', 'search']:
            self.model = build_model(args, self.data, self.device, self.method).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.loss_fn = F.cross_entropy
            batch_size_0 = 128 ### default
            ### node feature of all nodes in the graph
            ### all the neighbors of target node type in the graph
            self.neighbor_dict = self.data.neighbor_dict
            self.node_feat = {i: self.data.node_feat[i].to(self.device) for i in self.data.node_feat}

        ### set for architecture
        if self.method == 'search':
            self.neighbor_dict = self.data.neighbor_dict
            self.train_dict = {i:self.data.neighbor_dict[i] for i in self.train_idx.tolist()}
            self.valid_dict = {i:self.data.neighbor_dict[i] for i in self.valid_idx.tolist()}
            self.test_dict = {i:self.data.neighbor_dict[i] for i in self.test_idx.tolist()}
            self.loss_fn = F.cross_entropy
        self.no_increase = 0 ### design a early stopping

        if self.pretrain_model in ['RGCN']:
            if self.data.name =='OPPO':
                batch_size_0 = 1024 ### default
            else:
                batch_size_0 = 128
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.layer_num)
            self.train_loader_rgcn = dgl.dataloading.DataLoader(
                self.train_graph.to(self.device), {self.target: torch.tensor(self.data.split['seen']).to(self.device)}, sampler,
                batch_size=batch_size_0, device=self.device, shuffle=True, num_workers=0)#batch_size_0  len(self.data.split['seen'])
        
        self.save_path = self.init_model_path(args)
        self.val_max = 0

    def train(self):
        ######### pretrain start (added) #########
        if self.pretrain_model in ['GCN', 'GAT', 'GraphSAGE']:
            self.pre_train.edit_seen_graph()
            pre_feat_0 = self.pre_train.pretrain(total_epochs=1000) ### 1000
            pre_feat = {j:torch.tensor(pre_feat_0[j]).float().to(self.device) for j in pre_feat_0}
        else: ### the pretrain part for hetero ['RGCN']
            self.hetero_preprocess()
            epoch_num_rela = 15 #5 #pretrain ### 15
            for epoch in range(epoch_num_rela):
                metric, pre_loss = self.train_rela()
                if epoch % 3 == 0:
                    print('Epoch:{:d}: Train loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                                epoch, pre_loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
            if self.pretrain_model == 'RGCN': 
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.layer_num)
                if self.data.name == 'IMDB':
                    all_nodes_pre = self.data.split['seen'] + self.data.split['seen_wolbl']
                else:
                    all_nodes_pre = self.data.split['seen']
                out_feat_loader = dgl.dataloading.DataLoader(
                                self.train_graph.to(self.device), {self.target: torch.tensor(all_nodes_pre).to(self.device)}, sampler,
                                batch_size=len(all_nodes_pre), device=self.device, shuffle=True, num_workers=0)
                for i, (input_nodes, seeds, blocks) in enumerate(out_feat_loader):
                    blocks = [blk.to(self.device) for blk in blocks]
                    seeds = seeds[self.target] 
                    emb = extract_embed(self.train_model.input_feature(), input_nodes)
                    pre_feat_raw = self.train_model.forward_out_feat(blocks, emb)
                    pre_feat = {}
                    for j in self.train_graph.ntypes:
                        ### for unseen nodes, randomly generate the node feature
                        if self.data.name == 'DBLP' and j == 'conference':
                            continue
                        else:
                            # pre_feat[j] = torch.randn(self.data.nodenum_dict[j], self.args.in_dim).to(self.device)
                            pre_feat[j] = torch.tensor(np.random.randn(self.data.nodenum_dict[j], self.args.in_dim)).float().to(self.device)
                            pre_feat[j][input_nodes[j].tolist()] = pre_feat_raw[j]
        ######### pretrain end #########

        ### save node embedding
        torch.save({j:pre_feat[j].cpu() for j in pre_feat}, self.save_path + '/pre_dict.pkl')
        
        ### method train start
        if self.method in ['GCN', 'GraphSAGE', 'GAT', 'SingleModel']:
            # self.train_idx, self.valid_idx, self.test_idx split
            self.node_feat = pre_feat ### contain the pretrain embedding
            self.no_increase = 0
            for epoch in range(self.args.epoch_num):
                ### train
                metric, train_loss = self.train_step()
                print('epoch:'+ str(epoch))
                print('Train loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            train_loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
                if epoch % self.args.evaluate_interval == 0:
                    metric, loss = self.val_test_step('valid')
                    print('Validation loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
                    if metric['macro_f1'] > self.val_max: ### metric to be adapted
                        max_new = metric['macro_f1']
                        print(f'Validation macro_f1 increased ({self.val_max:.4f} --> {max_new:.4f}).  Saving model ...')
                        torch.save(self.model.state_dict(), self.save_path + '/meta_checkpoint.pkl')
                        self.val_max = max_new
                        self.no_increase = 0
                    else:
                        self.no_increase = self.no_increase + 1
                        if self.no_increase >=  self.args.epoch_num / 5:
                            print('Conduct the early stop step.')
                            break
            self.model_test = self.renew_model()
            self.model_test.load_state_dict(torch.load(self.save_path + '/meta_checkpoint.pkl'))
            self.model_test.eval()
            metric, loss = self.val_test_step('test')
        
        ### begin search process
        elif self.method in ['search']:
            res = []
            print('searched archs for {}...'.format(self.args.dataset))
            save_dir = '{}-{}'.format(self.args.dataset, time.strftime("%Y%m%d-%H%M%S"))
            save_dir = 'logs/search-{}'.format(save_dir)
            if not os.path.exists(save_dir):
                utils.create_exp_dir(save_dir, scripts_to_save=None) #changed
            log_filename = os.path.join(save_dir, 'log.txt')
            if not os.path.exists(log_filename):
                init_logger('', log_filename, logging.INFO, False)

            ### split data
            self.node_feat = pre_feat ### contain the pretrain embedding
            self.train_data = self.process_dict_to_data(self.train_dict, self.node_feat)
            self.valid_data = self.process_dict_to_data(self.valid_dict, self.node_feat)
            self.test_data = self.process_dict_to_data(self.test_dict, self.node_feat)
            if self.args.raw_feat:
                self.raw_feat = self.data.raw_feat.to(self.device)
                self.args.raw_dim = self.raw_feat.shape[1]
                pass

            seed = np.random.randint(0, 10000)
            self.args.seed = seed
            torch.cuda.set_device(self.args.cuda)
            cudnn.benchmark = True
            torch.manual_seed(self.args.seed)
            cudnn.enabled = True
            np.random.seed(self.args.seed)
            torch.cuda.manual_seed(self.args.seed)

            ### for circular
            self.model = build_model(self.args, self.data, self.device, self.method).to(self.device)
            self.model_optimizer = torch.optim.Adam(self.model.parameters(),self.args.lr,weight_decay=self.args.weight_decay)
            self.arch_optimizer = torch.optim.Adam(self.model.arch_parameters(),lr=self.args.arch_lr,weight_decay=self.args.arch_weight_decay)
            
            ## model scheduler
            self.model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, float(self.args.epoch_num), eta_min=self.args.lr_min)
            self.temp_scheduler = utils.Temp_Scheduler(int(self.args.epoch_num/2), self.args.temp, self.args.temp, temp_min=self.args.temp_min)
            
            logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))
            search_cost = 0
            for epoch in range(self.args.epoch_num):
                t1 = time.time()
                lr = self.model_scheduler.get_last_lr()[0]
                if self.args.cos_temp and epoch >= int(self.args.epoch_num/2):
                    self.model.temp = self.temp_scheduler.step()
                else:
                    self.model.temp = self.args.temp # 0.5
                metric, train_loss, valid_metric, valid_loss = self.train_step()
                self.model_scheduler.step()
                t2 = time.time()
                search_cost += (t2 - t1)
                print('epoch:'+ str(epoch))
                ### print train/valid
                print('Train loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            train_loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
                print('Valid loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            valid_loss, valid_metric['macro_f1'], valid_metric['micro_f1'], valid_metric['acc']))
                if (epoch + 1) % self.args.test_interval == 0:
                    test_metric, test_loss = self.val_test_step()
                    print('Test loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            test_loss, test_metric['macro_f1'], test_metric['micro_f1'], test_metric['acc']))
                genotype = self.model.genotype()
                logging.info('genotype = %s', genotype)
            logging.info('The search process costs %.2fs', search_cost)
            res.append('seed={},genotype={},saved_dir={},val_acc={},test_acc={}'.format(self.args.seed, genotype, save_dir, valid_metric['acc'], test_metric['acc']))

            ### single model
            return_model = genotype.split('||')
            self.method = 'SingleModel'
            self.args.method = 'SingleModel'
            arch_trans = return_model[:self.args.trans_num]
            arch_agg = return_model[self.args.trans_num]
            # self.data
            self.args.arch_dict = {'trans': arch_trans, 'agg_node': arch_agg, 'activate': 'relu'}
            self.model = build_model(self.args, self.data, self.device, self.method).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(),self.args.lr,weight_decay=self.args.weight_decay)

            self.node_feat = pre_feat ### contain the pretrain embedding
            self.no_increase = 0
            for epoch in range(100):
                ### train
                metric, train_loss = self.train_step()
                print('epoch:'+ str(epoch))
                print('Train loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            train_loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
                if epoch % self.args.evaluate_interval == 0:
                    metric, loss = self.val_test_step('valid')
                    print('Validation loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
                    if metric['macro_f1'] > self.val_max: ### metric to be adapted
                        max_new = metric['macro_f1']
                        print(f'Validation macro_f1 increased ({self.val_max:.4f} --> {max_new:.4f}).  Saving model ...')
                        torch.save(self.model.state_dict(), self.save_path + '/meta_checkpoint.pkl')
                        self.val_max = max_new
                        self.no_increase = 0
                    else:
                        self.no_increase = self.no_increase + 1
                        if self.no_increase >=  50:
                            print('Conduct the early stop step.')
                            break
            ### test step
            self.model_test = self.renew_model()
            self.model_test.load_state_dict(torch.load(self.save_path + '/meta_checkpoint.pkl'))
            self.model_test.eval()
            metric, loss = self.val_test_step('test')
            print('Final test loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
            print('Final supernet loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                            test_loss, test_metric['macro_f1'], test_metric['micro_f1'], test_metric['acc']))

            filename = 'exp_res/%s-searched_res-%s-reg%s.txt' % (self.args.dataset, time.strftime('%Y%m%d-%H%M%S'), self.args.weight_decay)
            fw = open(filename, 'w+')
            fw.write('\n'.join(res))
            fw.close()
            print('searched res for {} saved in {}'.format(self.args.dataset, filename))

        return metric['macro_f1']

    ### set of train function
    def train_step(self): # downstream tasks
        self.model.train()
        if self.args.method in ['RGCN']:
            loss = self.train_rela()
        elif self.args.method == 'Linear':
            loss = self.train_linear()
        elif self.args.method in ['GraphSAGE', 'GAT', 'GCN']:
            loss = self.tvt_kgrela('train')
        elif self.args.method in ['SingleModel']:
            loss = self.tvt_kgrela('train')
        elif self.args.method in ['search']:
            loss = self.train_search()
        else:
            pass
        return loss
    
    ### set of validation/test function
    def val_test_step(self, mode = 'test'):
        self.model.eval()
        with torch.no_grad():
            if mode == 'valid':
                num_for = self.valid_idx
                dict_for = self.data.valid_node_record
            else:
                num_for = self.test_idx
                dict_for = self.data.test_node_record
            if self.method in ['RGCN']:
                if self.inductive:
                    metric, loss_0 = self.valid_test_rela(num_for, dict_for) # use related works
                else:
                    metric, loss_0 = self.valid_test_rela_trans(num_for, dict_for, mode)
            elif self.method == 'Linear':
                metric, loss_0 = self.valid_test_linear(mode)
            elif self.args.method in ['GraphSAGE', 'GAT', 'GCN']:
                metric, loss_0 = self.tvt_kgrela(mode)
            elif self.args.method in ['SingleModel']:
                metric, loss_0 = self.tvt_kg_single(mode)
            elif self.args.method in ['search']:
                metric, loss_0 = self.test_search()
            else:
                pass
        return metric, loss_0

    def renew_model(self):
        if self.args.method in ['RGCN'] or (self.args.method == 'Ours' and self.args.use_heter_feature == True):
            new_model = build_model(self.args, self.data, self.device, self.method).to(self.device)
            data_to = {i:self.train_graph.ndata['h'][i].to(self.device) for i in self.train_graph.ndata['h']}
            input_feature = HeteroFeature(data_to, self.data.get_node_class_num(),
                self.args.in_dim, learn_feat_i=self.args.with_feature & self.args.learnable_feature).to(self.device)
            new_model.add_module('input_feature', input_feature)
        else:
            new_model = build_model(self.args, self.data, self.device, self.method).to(self.device)
        return new_model
    
    ### search training
    def train_search(self):
        self.model_optimizer.zero_grad()
        self.arch_optimizer.zero_grad()
        ### need to reprocess the data
        train_data = self.train_data
        valid_data = self.valid_data
        train_loss = 0
        logits_train, label_train, train_loss = self.func_for(train_data, self.train_idx)
 
        predict_train = logits_train.argmax(dim=1).to('cpu')
        metric_train = evaluate(label_train.to('cpu'), predict_train)
        train_loss = train_loss/len(self.train_idx.tolist())

        # update w
        train_loss.backward(retain_graph=True)
        self.model_optimizer.step()

        valid_loss = 0
        logits_valid, label_valid, valid_loss = self.func_for(valid_data, self.valid_idx)
        predict_valid = logits_valid.argmax(dim=1).to('cpu')
        metric_valid = evaluate(label_valid.to('cpu'), predict_valid)
        valid_loss = valid_loss/len(self.valid_idx.tolist())        

        # update alpha
        self.model_optimizer.zero_grad()
        self.arch_optimizer.zero_grad()
        valid_loss.backward(retain_graph=True)
        self.arch_optimizer.step()

        print(self.model.genotype())
        
        return metric_train, train_loss.item(), metric_valid, valid_loss.item()

    ### search testing
    def test_search(self):
        test_data = self.test_data
        logits_test, label_test, test_loss = self.func_for(test_data, self.test_idx)
        predict_test = logits_test.argmax(dim=1).to('cpu')
        metric_test = evaluate(label_test.to('cpu'), predict_test)
        test_loss = test_loss/len(self.test_idx.tolist())
        return metric_test, test_loss.item()  

    def process_dict_to_data(self, dict_0, data_0):
        data_dict = {}
        for i in dict_0:
            data_dict[i] = {}
            for j in dict_0[i]:
                data_dict[i][j] = data_0[j][dict_0[i][j]]
        return data_dict

    ### function for test
    def func_for(self, data, idx):
        logits_all = None
        label_all = None 
        flag_00 = 0
        loss_0 = 0
        label_all = self.labels[idx]
        for i in idx.tolist():
            if self.args.raw_feat:
                data_0 = [data[i], self.raw_feat[i]]
            else:
                data_0 = data[i]
            feat_00 = self.model(data_0).unsqueeze(0)
            if flag_00 == 0:
                logits_all = feat_00
                flag_00 = 1
            else:
                logits_all = torch.cat([logits_all, feat_00])
            # print('feat_00.shape: {}, self.labels[i].unsqueeze(0): {}'.format(feat_00.shape, self.labels[i].unsqueeze(0).shape))
            loss = self.loss_fn(feat_00, self.labels[i].unsqueeze(0).to(self.device))
            loss_0 += loss
        return logits_all, label_all, loss_0

    ### the train/valid/test of related works
    def tvt_kgrela(self, mode):
        loss_0 = 0.0
        logits_all = None
        label_all = None
        if mode == 'train':
            this_idx = self.train_idx
            this_record = self.data.train_node_record
        elif mode == 'valid':
            this_idx = self.valid_idx
            this_record = self.data.valid_node_record
        elif mode == 'test':
            this_idx = self.test_idx
            this_record = self.data.test_node_record
        flag_00 = 0
        label_all = self.labels[this_idx]
        if self.method in ['GraphSAGE', 'GAT', 'GCN']:
            for unseen in this_idx.tolist():
                link_record = this_record[unseen]
                this_feat = self.node_feat[self.target][unseen].unsqueeze(0)
                feat_all = [this_feat]
                idx_init = 1 ### unseen as 0
                for j in self.data.canonical_etypes:
                    if j[0] == self.target and j[1] in link_record:
                        link_0 = link_record[j[1]][1]
                        feat_0 = self.node_feat[j[2]][link_0]
                        feat_all.append(feat_0)
                        idx_init += len(link_0)
                edge_00 = torch.zeros(idx_init - 1).long()
                edge_11 = torch.tensor(np.array(list(set(range(1,idx_init))))).long()
                edge_final = torch.cat([torch.cat([edge_00,edge_11]).unsqueeze(0),torch.cat([edge_11,edge_00]).unsqueeze(0)],dim = 0).to(self.device)
                feat_final = torch.cat(feat_all).to(self.device)
                feat_00 = self.model(feat_final, edge_final) #筛选后的图
                if flag_00 == 0:
                    logits_all = feat_00
                    flag_00 = 1
                else:
                    logits_all = torch.cat([logits_all, feat_00])
                loss = self.loss_fn(feat_00, self.labels[unseen].unsqueeze(0).to(self.device))
                loss_0 += loss
        elif self.method in ['SingleModel']:
            for unseen in this_idx.tolist():
                neighbor_dict = self.neighbor_dict[unseen]
                feat_00 = self.model(self.node_feat[self.target][unseen].unsqueeze(0), neighbor_dict, self.node_feat).unsqueeze(0)
                if flag_00 == 0:
                    logits_all = feat_00
                    flag_00 = 1
                else:
                    logits_all = torch.cat([logits_all, feat_00])
                loss = self.loss_fn(feat_00, self.labels[unseen].unsqueeze(0).to(self.device))
                loss_0 += loss
        predict_all = logits_all.argmax(dim=1).to('cpu')
        metric = evaluate(label_all.to('cpu'), predict_all)
        loss_0 = loss_0/len(this_idx.tolist())
        if mode == 'train':
            self.optimizer.zero_grad()
            loss_0.backward(retain_graph=True)
            self.optimizer.step()
        return metric, loss_0.item()
    
    def tvt_kg_single(self, mode):
        loss_0 = 0.0
        logits_all = None
        label_all = None 
        if mode == 'train':
            this_idx = self.train_idx
        elif mode == 'valid':
            this_idx = self.valid_idx
        elif mode == 'test':
            this_idx = self.test_idx
        flag_00 = 0
        label_all = self.labels[this_idx]

        for unseen in this_idx.tolist():
            neighbor_dict = self.neighbor_dict[unseen]
            # 逐节点加入
            feat_00 = self.model(self.node_feat[self.target][unseen].unsqueeze(0), neighbor_dict, self.node_feat).unsqueeze(0)
            if flag_00 == 0:
                logits_all = feat_00
                flag_00 = 1
            else:
                logits_all = torch.cat([logits_all, feat_00])
            loss = self.loss_fn(feat_00, self.labels[unseen].unsqueeze(0).to(self.device))
            loss_0 += loss
                
        predict_all = logits_all.argmax(dim=1).to('cpu')
        metric = evaluate(label_all.to('cpu'), predict_all)
        loss_0 = loss_0/len(this_idx.tolist())
        if mode == 'train':
            self.optimizer.zero_grad()
            loss_0.backward(retain_graph=True)
            self.optimizer.step()
        return metric, loss_0.item()

    ### to be refered, how to get 
    def train_rela(self): #use pretrain model 
        loss_0 = 0.0
        logits_all = None
        label_all = None
        if self.pretrain_model in ['RGCN']:
            for i, (input_nodes, seeds, blocks) in enumerate(self.train_loader_rgcn):
                blocks = [blk.to(self.device) for blk in blocks]
                seeds = seeds[self.target] 
                emb = extract_embed(self.train_model.input_feature(), input_nodes)
                # seeds = seeds.to(self.device) # added
                # print(f'seeds.device: {seeds.device}, self.labels.device: {self.labels.device}')
                self.labels = self.labels.to(self.device) # added
                lbl = self.labels[seeds].to(self.device)
                logits = self.train_model(blocks, emb)[self.target]
                loss = self.train_loss_fn(logits, lbl)
                # loss = self.train_loss_fn(F.softmax(logits, dim=1), lbl)
                loss_0 += loss
                self.train_optimizer.zero_grad()
                loss.backward()
                self.train_optimizer.step()
                torch.cuda.empty_cache()
                if logits_all == None:
                    logits_all = logits
                    label_all = lbl
                else:
                    logits_all = torch.cat([logits_all, logits], dim = 0)
                    label_all = torch.cat([label_all, lbl], dim = 0)
            loss_0 = loss_0 / (i + 1)
        predict_all = logits_all.argmax(dim=1).to('cpu')
        label_all = label_all.cpu()
        metric = evaluate(label_all, predict_all)
        return metric, loss_0.item()

    def hetero_preprocess(self):
        if self.args.with_feature:
            data_to = {i:self.train_graph.ndata['h'][i].to(self.device) for i in self.train_graph.ndata['h']}
            self.input_feature = HeteroFeature(data_to, self.data.get_node_class_num(),
                self.args.in_dim, learn_feat_i=self.args.with_feature & self.args.learnable_feature).to(self.device)
            self.train_optimizer.add_param_group({'params': self.input_feature.parameters()})
            self.train_model.add_module('input_feature', self.input_feature)
        else:
            data_to = {i:self.train_graph.ndata['h'][i].to(self.device) for i in self.train_graph.ndata['h']}
            self.input_feature = HeteroFeature(data_to, self.data.get_node_class_num(),
                                                    self.args.in_dim).to(self.device)                
            self.train_optimizer.add_param_group({'params': self.input_feature.parameters()})
            self.train_model.add_module('input_feature', self.input_feature)

    # @profile
    def valid_test_rela(self, num_for, dict_for):
        logits_all = None
        label_all = None
        metric, loss_0 = 0.0, 0.0
        ii = 0
        for j in num_for:
            j = j.item()
            # graph_copy = copy.deepcopy(self.train_graph)
            for k in dict_for[j]:
                self.train_graph = dgl.add_edges(self.train_graph, dict_for[j][k][0], dict_for[j][k][1], etype = k)
            if self.method in ['RGCN']:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.layer_num)
                input_nodes, seeds, blocks = sampler.sample_blocks(self.train_graph.to(self.device), 
                                   {self.target: torch.tensor([j]).to(self.device)})
                blocks = [blk.to(self.device) for blk in blocks]
                seeds = seeds[self.target]
                emb = extract_embed(self.model.input_feature(), input_nodes)
                lbl = self.labels[seeds].to(self.device)
                logits = self.model(blocks, emb)[self.target]
                loss = self.loss_fn(logits, lbl)
            loss_0 += loss.item()
            if logits_all == None:
                logits_all = logits
                label_all = lbl
            else:
                logits_all = torch.cat([logits_all, logits], dim = 0)
                label_all = torch.cat([label_all, lbl], dim = 0)
            ii = ii + 1
            for k in dict_for[j]:
                edge_num_now = self.train_graph.num_edges(etype = k)
                self.train_graph = dgl.remove_edges(self.train_graph, list(range(edge_num_now - len(dict_for[j][k][0]), edge_num_now)), etype = k)
        loss_0 = loss_0/ii
        predict_all = logits_all.argmax(dim=1).to('cpu')
        label_all = label_all.cpu()
        metric = evaluate(label_all, predict_all)
        return metric, loss_0

    ### set save path
    def init_model_path(self, args):
        result_path = './results/' + args.dataset + '_' + args.method + '_' + str(args.epoch_num) + '_epochs'
        os.makedirs(result_path, exist_ok=True)
        trial_id = 0
        path_exists = True
        while path_exists:
            trial_id += 1
            path_to_results = result_path + '/{:d}'.format(trial_id)
            path_exists = os.path.exists(path_to_results)
        os.makedirs(path_to_results)
        return path_to_results
    
    ### training for linear function(not used now)
    def train_linear(self): ### the learn process of linear
        i = 0
        loss_0 = 0.0 
        logits_all = None
        label_all = None
        for data in self.train_loader:
            feature = data[:,:-1]
            lbl = data[:,-1].long()
            logits = self.model(feature)
            loss = self.loss_fn(logits, lbl)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_0 = loss_0 + loss.item()
            i = i + 1
            if logits_all == None:
                logits_all = logits
                label_all = lbl
            else:
                logits_all = torch.cat([logits_all, logits], dim = 0)
                label_all = torch.cat([label_all, lbl], dim = 0)
        predict_all = logits_all.argmax(dim=1).to('cpu')
        label_all = label_all.cpu()
        metric = evaluate(label_all, predict_all)
        return metric, loss_0 / i

    def valid_test_linear(self, mode):
        logits_all = None
        label_all = None
        metric, loss_0 = 0.0, 0.0
        if mode == 'valid':
            data_load = self.valid_loader
        else:
            data_load = self.test_loader
        i = 0
        for data in data_load:
            feature = data[:,:-1]
            lbl = data[:,-1].long()
            logits = self.model(feature)
            loss = self.loss_fn(logits, lbl)  
            loss_0 += loss.item()          
            if logits_all == None:
                logits_all = logits
                label_all = lbl
            else:
                logits_all = torch.cat([logits_all, logits], dim = 0)
                label_all = torch.cat([label_all, lbl], dim = 0)
            i = i + 1
        predict_all = logits_all.argmax(dim=1).to('cpu')
        label_all = label_all.cpu() 
        metric = evaluate(label_all, predict_all)
        return metric, loss_0/i

    def preprocess_wfeat(self):
        if self.args.method in ['RGCN'] or (self.args.method == 'Ours' and self.args.use_heter_feature == True):
            data_to = {i:self.train_graph.ndata['h'][i].to(self.device) for i in self.train_graph.ndata['h']}
            self.input_feature = HeteroFeature(data_to, self.data.get_node_class_num(),
                self.args.in_dim, learn_feat_i=self.args.with_feature & self.args.learnable_feature).to(self.device)
            self.optimizer.add_param_group({'params': self.input_feature.parameters()})
            self.model.add_module('input_feature', self.input_feature)
        return

    def preprocess_wofeat(self):
        if self.args.method in ['RGCN']:
            data_to = {i:self.train_graph.ndata['h'][i].to(self.device) for i in self.train_graph.ndata['h']}
            self.input_feature = HeteroFeature(data_to, self.data.get_node_class_num(),
                                                    self.args.in_dim).to(self.device)                
            self.optimizer.add_param_group({'params': self.input_feature.parameters()})
            self.model.add_module('input_feature', self.input_feature)
        return

    def valid_test_rela_trans(self, num_for, dict_for, mode): ### an experiment for transductive setting
        logits_all = None
        label_all = None
        metric, loss_0 = 0.0, 0.0
        if self.method in ['RGCN']:
            if mode == 'valid':
                my_loader = self.valid_loader
            else:
                my_loader = self.test_loader
            for i, (input_nodes, seeds, blocks) in enumerate(my_loader):
                blocks = [blk.to(self.device) for blk in blocks]
                seeds = seeds[self.target]
                emb = extract_embed(self.model.input_feature(), input_nodes)
                lbl = self.labels[seeds].to(self.device)
                logits = self.model(blocks, emb)[self.target]
                loss = self.loss_fn(logits, lbl)
                loss_0 += loss.item()
                if logits_all == None:
                    logits_all = logits
                    label_all = lbl
                else:
                    logits_all = torch.cat([logits_all, logits], dim = 0)
                    label_all = torch.cat([label_all, lbl], dim = 0)
            loss_0 = loss_0/(i+1)
        predict_all = logits_all.argmax(dim=1).to('cpu')
        label_all = label_all.cpu()
        metric = evaluate(label_all, predict_all)
        return metric, loss_0
    
    
### build model, to call the function of the each model
def build_model(args, data, device, method, with_feat = False):
    if with_feat:
        dim_this = args.in_dim
    else:
        dim_this = data.in_dim
    model = 0
    if method == 'RGCN': ### set num_bases as 1, do not understand its function
        model = RGCN(args.in_dim, args.in_dim, data.num_classes, data.edge_type, -1, args.layer_num - 2, args.dropout)
    elif method == 'GraphSAGE':
        model = SAGE_model(args.in_dim, args.in_dim, data.num_classes)
    elif method == 'GAT':
        model = GAT_model(args.in_dim, args.in_dim, data.num_classes)
    elif method == 'GCN':
        model = GCN_model(args.in_dim, args.in_dim, data.num_classes)
    elif method == 'SingleModel':
        arch_dict = args.arch_dict
        model = SingleModel(device, args.in_dim, args.in_dim, data.num_classes, arch_dict, [i for i in data.neighbor_dict[0]])
    elif method == 'search':
        model = Search_Network(device, args.in_dim, args.in_dim, data.num_classes, args, [i for i in data.neighbor_dict[0]])
    else:
        pass
    return model

def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb

### evaluation metrics calculation
def evaluate(label_all, predict_all):
    macro_f1 = f1_score(label_all, predict_all, average='macro')
    micro_f1 = f1_score(label_all, predict_all, average='micro')   
    acc = accuracy_score(label_all, predict_all)   
    metric = {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'acc': acc}
    return metric

def get_nodes_with_edges(graph):
    nd_dict = {j:[] for j in graph.ntypes}
    can_list = {graph.to_canonical_etype(i):graph.edges(etype = i) for i in graph.etypes}
    for j in can_list:
        if can_list[j][0].shape[0] > 0:
            list_0 = can_list[j][0].tolist() ### might change to tensor
            list_1 = can_list[j][1].tolist()
            for i in list_0:
                if i in nd_dict[j[0]]:
                    pass
                else:
                    nd_dict[j[0]].append(i)
            for i in list_1:
                if i in nd_dict[j[2]]:
                    pass
                else:
                    nd_dict[j[2]].append(i)
        else:
            continue
    return nd_dict

### add reverse edges
def add_reverse(graph):
    for type0 in graph.canonical_etypes:
        if graph.edges(etype = type0[1])[0].shape[0] > graph.edges(etype = type0[2] + '-' + type0[0])[0].shape[0]:
            pass
        else:
            edge_need = graph.edges(etype = type0[2] + '-' + type0[0])
            graph = dgl.add_edges(graph, edge_need[1], edge_need[0], etype = type0[1])
    return graph

