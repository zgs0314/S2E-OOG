import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processor import NodeClassDataset
from models.rela_model import SingleModel
from sklearn.metrics import f1_score, accuracy_score

class NC_Evaluator(nn.Module):
    def __init__(self, args):
        super(NC_Evaluator, self).__init__()
        self.args = args
        self.data = NodeClassDataset(args)
        self.train_idx, self.valid_idx, self.test_idx = self.data.get_split()
        self.labels = self.data.get_labels()
        self.neighbor_dict = self.data.neighbor_dict
        self.loss_fn = F.cross_entropy
        self.target = self.data.target_ntype
        self.device = torch.device("cuda:"+ str(args.cuda) if torch.cuda.is_available() else "cpu")
        if args.dataset == 'ACM':
            self.args.arch_dict = {'trans': ['identity', 'rel-linear'], 'agg_node': 'mean', 'activate': 'relu'}
        elif args.dataset == 'IMDB':
            self.args.arch_dict = {'trans': ['rel-linear', 'identity'], 'agg_node': 'mean', 'activate': 'relu'}
        elif args.dataset == 'OPPO':
            self.args.arch_dict = {'trans': ['rel-linear', 'linear', 'rel-linear'], 'agg_node': 'mean', 'activate': 'relu'}
        # self.node_feat
        self.node_feat = torch.load('model_file/pre_emb/pre_dict_' + self.args.dataset + '.pkl')
        self.node_feat = {j:self.node_feat[j].to(self.device) for j in self.node_feat}

    ### evaluat for the model performance
    def evaluate(self):
        self.model_test = self.build_model(self.args, self.data, self.device).to(self.device)
        self.model_test.load_state_dict(torch.load('model_file/final_model_' + self.args.dataset + '.pkl'))
        self.model_test.eval()
        metric, loss = self.test_step()
        print('Final test loss:{:.4f}, macro_f1:{:.4f}, micro_f1:{:.4f}, accuracy:{:.4f}'.format(
                        loss, metric['macro_f1'], metric['micro_f1'], metric['acc']))
        return

    ### build model, to call the function of the each model
    def build_model(self, args, data, device):
        arch_dict = args.arch_dict
        model = SingleModel(device, args.in_dim, args.in_dim, data.num_classes, arch_dict, [i for i in data.neighbor_dict[0]])
        return model
    
    ### examine each sample in the test dataset
    def test_step(self):
        this_idx = self.test_idx
        label_all = self.labels[this_idx]
        flag_00 = 0
        loss_0 = 0
        for unseen in this_idx.tolist():
            neighbor_dict = self.neighbor_dict[unseen]
            feat_00 = self.model_test(self.node_feat[self.target][unseen].unsqueeze(0), neighbor_dict, self.node_feat).unsqueeze(0)
            if flag_00 == 0:
                logits_all = feat_00
                flag_00 = 1
            else:
                logits_all = torch.cat([logits_all, feat_00])
            loss = self.loss_fn(feat_00, self.labels[unseen].unsqueeze(0).to(self.device))
            loss_0 += loss
                
        predict_all = logits_all.argmax(dim=1).to('cpu')
        metric = self.evaluate_0(label_all.to('cpu'), predict_all)
        loss_0 = loss_0/len(this_idx.tolist())
        return metric, loss_0.item()

    ### evaluation metrics calculation
    def evaluate_0(self, label_all, predict_all):
        macro_f1 = f1_score(label_all, predict_all, average='macro')
        micro_f1 = f1_score(label_all, predict_all, average='micro')   
        acc = accuracy_score(label_all, predict_all)   
        metric = {'macro_f1': macro_f1, 'micro_f1': micro_f1, 'acc': acc}
        return metric