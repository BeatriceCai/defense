import argparse
# import random
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
import torch
# --------------
from data_tool_box_DRD import str2bool,save_json,set_up_exp_folder,load_pkl,save_dict_pickle
from utils_DRD import sample_minibatch_maml, sample_minibatch_classic,sample_minibatch_test
from DTI_meta_MAML_DRD import meta_DTI_MAML
# -------------------------------
parser = argparse.ArgumentParser("DTI in a MAML way: TRAINED FOR DRD ")
# ... # ...admin # ...# ...# ...
parser.add_argument('--use_cuda',type=str2bool, nargs='?',const=True, default=True, help='use cuda.')
# parser.add_argument('--fr_scratch',type=str2bool, default=False)
parser.add_argument('--cwd', type=str,  default='',
                    help='define your own current working directory,i.e. where you put your scirpt')
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
# ... # ...meta core # ...# ...# ...
parser.add_argument('--k_spt', default=1, type=int, help='k-shot learning')
parser.add_argument('--k_qry', default=4,type=int)
parser.add_argument('--task_num',default=5,type=int)
parser.add_argument('--mix_n',default=1,type=int)
parser.add_argument('--meta_lr',default=1e-3,type=float)
parser.add_argument('--update_lr',default=0.01,type=float)
parser.add_argument('--update_step',default=5,type=int)
parser.add_argument('--update_step_test',default=10,type=int)
# ... # ...model # ...# ...# ...
parser.add_argument('--frozen',default='none',type=str)
parser.add_argument('--phase1ResiDist', type=str, default='none', help = {'multiply-binary','none'})
parser.add_argument('--phase2DTIDist', type=str, default='multiply-multi', help = {'multiply-binary','none'})
# ... # ...global optimization # ...# ...# ...
parser.add_argument('--test_batch_size_pos', default=12, type=int)
parser.add_argument('--test_batch_size_neg', default=64, type=int)
parser.add_argument('--classic_batch_size', default=32, type=int)
parser.add_argument('--global_MAML_step', default=20, type=int,
                    help='Number of global training steps, i.e. numberf of mini-batches ')
parser.add_argument('--classic_at', default=5, type=int, help='')
parser.add_argument('--global_eval_at', default=10, type=int, help='')
parser.add_argument('--global_eval_step',default=100, type=int)
opt = parser.parse_args()
# -------------------------------
args = {}
args.update(vars(opt))
DTI_classifier_config ={
    'chem_pretrained':'nope',
    'protein_descriptor':'DISAE',
    'DISAE':{'frozen_list':[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]}
}
args.update(DTI_classifier_config)
seed = 705
np.random.seed(args['seed'])
print(f"split: {args['split']}, seed: {args['seed']}")
torch.manual_seed(seed)
if opt.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    args['device']='cuda'
else:
    args['device'] = 'cpu'
checkpoint_path = set_up_exp_folder(args['cwd'],exp ='exp_DRD_portalCG_logs/')
config_file = checkpoint_path + 'config.json'
save_json(vars(opt),    config_file)
# -------------------------------
if __name__ == '__main__':
    # load data
    MAML_DTI_path = args['cwd'] + 'data/DRD/'
    train_classic = load_pkl(MAML_DTI_path + f"split-{args['split']}/ood_train_classic_domained.pkl")
    train_maml = load_pkl(MAML_DTI_path +  f"split-{args['split']}/ood_train_maml_domained.pkl")
    ood_test = load_pkl(MAML_DTI_path + f"split-{args['split']}/ood_test_domained.pkl")
    train_maml_pfam = list(train_maml.keys())
    # set up meta core
    maml = meta_DTI_MAML(args)

    train_maml_performance ={'loss':[],'overall':[],'class0':[],'class1':[]}
    train_classic_performance = {'loss': [], 'overall': [], 'class0': [], 'class1': []}
    ood_test_performance ={'loss':[],'overall':[],'class0':[],'class1':[]}
    best_test_AUC = -np.inf
    best_test_AUPR = -np.inf
    print(f"split: {args['split']}, seed: {args['seed']}")
    stime = time.time()
    print('\tF1\tROC-AUC\tPR-AUC')
    for step in range(args['global_MAML_step']):
        maml.train()
        batch_data = sample_minibatch_maml(args, train_maml_pfam, train_maml)
        losses, overall,class0,class1 = maml(batch_data)
        train_maml_performance['loss'].append(losses.detach().cpu().item())
        train_maml_performance['overall'].append(overall)
        train_maml_performance['class0'].append(class0)
        train_maml_performance['class1'].append(class1)
        if step % args['classic_at']==0:
            batch_data_classic = sample_minibatch_classic(args, train_classic)
            loss, overall, class0, class1 = maml.classic_train(batch_data_classic)
            train_classic_performance['loss'].append(loss.detach().cpu().item())
            train_classic_performance['overall'].append(overall)
            train_classic_performance['class0'].append(class0)
            train_classic_performance['class1'].append(class1)
        # -------------------------------------------
        #         evaluating
        # -------------------------------------------
        if step % args['global_eval_at'] ==0:
            print('------------------------training step: ', step)
            maml.eval()
            # -------------------------------------------
            #         zero-shot test
            # -------------------------------------------

            overall_ood_test, class0_ood_test, class1_ood_test, loss_ood_test = [], [], [], []
            for step_eval in range(args['global_eval_step']):
                batch_data_test = sample_minibatch_test(args, ood_test)
                # batch_data_ood_test = ood_test.sample(args['k_qry'] * 2 * args['mix_n'])
                # if len(set(batch_data_ood_test['Activity'].values.tolist())) > 1:
                losses, overall, class0, class1 = maml.zeroshot_test(batch_data_test)
                overall_ood_test.append(overall)
                class0_ood_test.append(class0)
                class1_ood_test.append(class1)
                loss_ood_test.append(losses.detach().cpu().item())

            ood_test_performance['loss'].append(np.mean(loss_ood_test))
            ood_test_performance['overall'].append(np.array(overall_ood_test).mean(axis=0))
            ood_test_performance['class0'].append(np.array(class0_ood_test).mean(axis=0))
            ood_test_performance['class1'].append(np.array(class1_ood_test).mean(axis=0))


            # -------------------------------------------
            #         save records
            # -------------------------------------------

            save_dict_pickle(train_maml_performance,checkpoint_path  +'train_maml_performance.pkl')
            save_dict_pickle(train_classic_performance, checkpoint_path + 'train_classic_performance.pkl')
            save_dict_pickle(ood_test_performance, checkpoint_path + 'ood_test_performance.pkl')
            print('train maml\t', train_maml_performance['overall'][-1])
            print('ood_test\t', ood_test_performance['overall'][-1])
            print(f' time cost {time.time() - stime}')
            stime = time.time()
            if ood_test_performance['overall'][-1][1] > best_test_AUC :
                best_test_AUC = ood_test_performance['overall'][-1][1]
                torch.save(maml.model.state_dict(), checkpoint_path+'maml_model.dat')
                print('..................saved at:', checkpoint_path)
            elif ood_test_performance['overall'][-1][2] > best_test_AUPR:
                best_test_AUPR =ood_test_performance['overall'][-1][2]
                torch.save(maml.model.state_dict(), checkpoint_path + 'maml_model.dat')
                print('..................saved at:', checkpoint_path)

print('training finished ~')
print(f'model performance record saved to {checkpoint_path}')