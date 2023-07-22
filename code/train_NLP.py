import argparse
# import random
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch import optim
# --------------
from data_tool_box_DRD import str2bool,save_json,set_up_exp_folder,load_pkl,save_dict_pickle
from utils_DRD import get_repr_DTI
from models_MAML import DTI_model_MAML

from NLPfunctions import *
from NLPmodels import *
from NLPutils import *
from NLP_data_utils import *
# -------------------------------
parser = argparse.ArgumentParser("DTI in a MAML way: TRAINED FOR DRD ")
# ... # ...admin # ...# ...# ...
parser.add_argument('--use_cuda',type=str2bool, nargs='?',const=True, default=True, help='use cuda.')
parser.add_argument('--cwd', type=str,  default='../',
                    help='define your own current working directory,i.e. where you put your scirpt')
parser.add_argument('--seed', default=1, type=int)
# ... # ...meta core # ...# ...# ...
parser.add_argument('--batch_size',default=48,type=int)
parser.add_argument('--val_range',default=1,type=int)
parser.add_argument('--meta_lr',default=1e-3,type=float)
parser.add_argument('--global_MAML_step', default=200000, type=int,
                    help='Number of global training steps, i.e. numberf of mini-batches ')
parser.add_argument('--global_eval_at', default=1000, type=int, help='')
opt = parser.parse_args()
# -------------------------------
args = {}
args.update(vars(opt))

seed = 705
np.random.seed(args['seed'])
# print(f"split: {args['split']}, seed: {args['seed']}")
torch.manual_seed(seed)
if opt.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    args['device']='cuda'
else:
    args['device'] = 'cpu'
checkpoint_path = set_up_exp_folder(args['cwd'],exp ='exp_NLP_logs/')
config_file = checkpoint_path + 'config.json'
save_json(vars(opt),    config_file)
# -------------------------------
if __name__ == '__main__':
    # load data
    V = 15
    padding_idx = 0

    train = pd.read_pickle('../data/train_tokenized.pkl')
    chem_dict = pd.Series(load_pkl(args['cwd'] + 'data/ChEMBLE26/chemical/ikey2smiles_ChEMBLE.pkl'))
    protein_dict = pd.Series(load_pkl(args['cwd'] + 'data/ChEMBLE26/protein/unipfam2triplet.pkl'))
    model = DTI_model_MAML(all_config=args).to('cuda')
    tokenizer = model.prot_tokenizer
    criterion_smooth = LabelSmoothing(V, padding_idx, smoothing=0.0)
    meta_optim = optim.Adam(model.parameters(), lr=args['meta_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optim, T_max=10)
    # train_maml_performance ={'loss':[],'overall':[],'class0':[],'class1':[]}
    train_classic_performance = {'loss': [], 'overall': []}

    stime = time.time()
    # print('\tF1\tROC-AUC\tPR-AUC')
    for step in range(args['global_MAML_step']):
        print(step)
        model.train()
        batch = train.sample(args['batch_size'])
        chem_graph, protein_tokenized = get_repr_DTI(batch, tokenizer, chem_dict, protein_dict, 'DISAE')
        # ---------------------
        out, ntokens, target = model.encode(protein_tokenized.to('cuda'),
                                            chem_graph.to('cuda'),
                                            batch)

        out_G = model.generator(out)
        loss = criterion_smooth(out_G.contiguous().view(-1, out_G.size(-1)), target.contiguous().view(-1))
        loss = loss / ntokens
    #     ---------------------
        meta_optim.zero_grad()
        loss.backward()
        meta_optim.step()
        scheduler.step()

        train_classic_performance['loss'].append(loss.detach().cpu().item())

        #         evaluating
        # -------------------------------------------
        if step % args['global_eval_at'] ==0:

            save_dict_pickle(train_classic_performance, checkpoint_path + 'train_classic_performance.pkl')

            torch.save(model.state_dict(), checkpoint_path+'NLP_model.dat')

save_dict_pickle(train_classic_performance, checkpoint_path + 'train_classic_performance.pkl')
torch.save(model.state_dict(), checkpoint_path+'NLP_model.dat')
print('training finished ~')
print(f'model performance record saved to {checkpoint_path}')