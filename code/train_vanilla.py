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
# from utils_DRD import sample_minibatch_maml, sample_minibatch_classic,sample_minibatch_test
from models_MAML import DTI_model_MAML
# from transforms import FileTransform, TempReverseTransform,correlationReverseTransform
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
checkpoint_path = set_up_exp_folder(args['cwd'],exp ='exp_vanilla_logs/')
config_file = checkpoint_path + 'config.json'
save_json(vars(opt),    config_file)
# -------------------------------
if __name__ == '__main__':
    # load data
    train = pd.read_csv('../data/train_set.csv')
    test = pd.read_csv('../data/test_set.csv')
    chem_dict = pd.Series(load_pkl(args['cwd'] + 'data/ChEMBLE26/chemical/ikey2smiles_ChEMBLE.pkl'))
    protein_dict = pd.Series(load_pkl(args['cwd'] + 'data/ChEMBLE26/protein/unipfam2triplet.pkl'))

    model = DTI_model_MAML(all_config=args).to('cuda').double()
    criterion = nn.MSELoss(reduction='mean')
    # loss_fn = torch.nn.CrossEntropyLoss()
    tokenizer = model.prot_tokenizer
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
        logit = model(protein_tokenized.to('cuda'),chem_graph.to('cuda'))
        loss = criterion(logit.cpu(), torch.tensor(batch['ic_value'].values))

        meta_optim.zero_grad()
        loss.backward()
        meta_optim.step()
        scheduler.step()

        train_classic_performance['loss'].append(loss.detach().cpu().item())

        #         evaluating
        # -------------------------------------------
        if step % args['global_eval_at'] ==0:

            save_dict_pickle(train_classic_performance, checkpoint_path + 'train_classic_performance.pkl')

            torch.save(model.state_dict(), checkpoint_path+'vanilla_model.dat')

save_dict_pickle(train_classic_performance, checkpoint_path + 'train_classic_performance.pkl')
torch.save(model.state_dict(), checkpoint_path+'vanilla_model.dat')
print('training finished ~')
print(f'model performance record saved to {checkpoint_path}')