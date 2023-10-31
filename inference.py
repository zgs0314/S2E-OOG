import torch
import numpy as np
import argparse
import random
import os
from evaluator import NC_Evaluator
import torch.backends.cudnn as cudnn

import setproctitle
print('pid:', os.getpid())

torch.set_num_threads(8)

def inference(args):
    print(args.dataset) # temporal
    evaluator = NC_Evaluator(args)
    evaluator.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', type=str, default='OPPO', help='Dataset to use, including, ACM, IMDB, OPPO.') 
    ### OPPO could only be tested under the setting "with_feature == False"
    parser.add_argument('--inductive', type=bool, default=True, help='Whether use the inductive setting') # True, False
    ### Now use the pretrain feature
    parser.add_argument('--with_feature', type=bool, default=False, help='Whether the nodes in the graph have initial feature') # True, False
    parser.add_argument('--learnable_feature', type=bool, default=False, help='Whether the nodes in the graph have initial feature') # now do not need to change
    parser.add_argument('-e','--epoch_num', type=int, default=50, help="Choose the epoch number") 
    parser.add_argument("--in_dim", type=int, default=128, help="The dimension of inner layer") 
    parser.add_argument("--layer_num", type=int, default=2, help="The layer numbers") 
    parser.add_argument("--dropout", type=float, default=0.05, help="The dropout rate(mainly related works)")
    ### hyperparameters about ours
    parser.add_argument('--edge_feature', type=bool, default=False, help='Whether use edge feature to propagate knowledge.') # True, False
    parser.add_argument('--use_heter_feature', type=bool, default=False, help='Whether use heterofeature model.') # True, False
    parser.add_argument('--ntype_layer', type=bool, default=True, help='Whether use different weight for agg for different node type.') # True, False
    parser.add_argument('--reg_sim', type=str, default='mlp', help='Choose the approach to calculate vector similarity, cos, mlp.')
    parser.add_argument("--reg_gamma", type=float, default=0.5, help="The dropout rate(mainly related works)")
    ### about UTIL
    parser.add_argument('--cuda', type=int, default=2, help='choose the GPU to do the training')
    parser.add_argument('--seed', type=int, default=2, help='choose the seed')
    ### pretrain
    # parser.add_argument("--use_pretrain", type=bool, default=False, help='whether use pretrain model')
    parser.add_argument("--pretrain_model", type=str, default='RGCN', help='pretrain backbone: GCN, GAT, GraphSAGE, RGCN') ### RGCN
    parser.add_argument("--pretrain_with_feature", type=bool, default=True, help='use raw feature as input feature for pretrain model') ### keep that true.
    parser.add_argument("--het_prelr", type=float, default=0.01, help='the learning rate of the pretrain heterogeneous model')
    parser.add_argument("--het_prewd", type=float, default=5e-5, help='the weight decay of the pretrain heterogeneous model')
    ### darts search related hyperparameter
    parser.add_argument('-t','--trans_num', type=int, default=3, help='number of transition operation layer')
    parser.add_argument('--test_interval', type=int, default=10, help='the epoch interval for test step')
    parser.add_argument('--raw_feat', type=bool, default=False, help='whether use raw feature for unseen nodes')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    setproctitle.setproctitle('Ablation#{}'.format(args.dataset))
    print(args)
    inference(args)
