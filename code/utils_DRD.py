import pandas as pd
import random
from sklearn import metrics
# --------------
from transformers import BertTokenizer
from transformers.configuration_albert import AlbertConfig
from transformers.modeling_albert import AlbertForMaskedLM
from transformers.modeling_albert import load_tf_weights_in_albert
# --------------
from rdkit import Chem
from torch_geometric.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch
import torch
from typing import Sequence, Tuple, List, Union

from ligand_graph_features import *
from data_tool_box_DRD import load_json

def sample_minibatch_maml(args,train_maml_pfam,train_maml):
    batch_data_maml = []
    batch_pfam_maml = np.random.choice(train_maml_pfam,args['task_num'])


    for task_pfam in batch_pfam_maml:
    # for task_pfam in train_maml_pfam:
        per_task ={}
        task_data = train_maml[task_pfam]
        for s in ['spt','qry']:
            df = pd.DataFrame()
            df= df.append(  task_data['pos'].sample(args['k_' +s], replace=True))
            df= df.append(  task_data['neg'].sample(args['k_'+s], replace=True))
            per_task[s]= df
        batch_data_maml.append(per_task)
    return batch_data_maml

def sample_minibatch_classic(args,train_classic):
    batch_pos = train_classic['pos'].sample(args['classic_batch_size'])
    batch_neg = train_classic['neg'].sample(args['classic_batch_size'])

    batch_classic = pd.concat([batch_pos, batch_neg])
    return batch_classic
def sample_minibatch_test(args,train_classic):
    batch_pos = train_classic['pos'].sample(args['test_batch_size_pos'])
    batch_neg = train_classic['neg'].sample(args['test_batch_size_neg'])

    batch_classic = pd.concat([batch_pos, batch_neg])
    return batch_classic
# =======================================================
#             model related
# =======================================================
def load_DISAE(cwd):
    all_config = load_json( 'DTI_config.json')
    albertconfig = AlbertConfig.from_pretrained(cwd+ all_config['DISAE']['albertconfig'])
    m = AlbertForMaskedLM(config=albertconfig)
    m = load_tf_weights_in_albert(m, albertconfig, cwd+all_config['DISAE']['albert_pretrained_checkpoint'])
    prot_descriptor = m.albert
    prot_tokenizer = BertTokenizer.from_pretrained(cwd+all_config['DISAE']['albertvocab'])
    return  prot_descriptor,prot_tokenizer
# =======================================================
#             maml
# =======================================================
def get_repr_DTI_MAML(batch_data_pertask,mode,prot_tokenizer,chem_dict,protein_dict,args):
    batch_chem_graphs_spt,batch_protein_tokenized_spt = get_repr_DTI(batch_data_pertask[mode],
                                                                    prot_tokenizer,
                                                                    chem_dict,protein_dict,
                                                                    'DISAE')
    batch_chem_graphs_spt = batch_chem_graphs_spt.to(args['device'])
    batch_protein_tokenized_spt = batch_protein_tokenized_spt.to(args['device'])

    y_spt = torch.LongTensor(list(batch_data_pertask[mode]['Activity'].values)).to(args['device'])
    return batch_chem_graphs_spt,batch_protein_tokenized_spt,y_spt

def core_batch_prediction_MAML(batch_data_pertask, args, tokenizer,mode,loss_fn,
                               chem_dict, protein_dict, model,detach=False):

    batch_chem_graphs_spt, batch_protein_tokenized_spt, y_spt = get_repr_DTI_MAML(batch_data_pertask,
                                                              mode,
                                                              tokenizer, chem_dict, protein_dict,
                                                              args)

    batch_logits = model(batch_protein_tokenized_spt, batch_chem_graphs_spt)
    loss = loss_fn(batch_logits,y_spt)
    if detach == True:
        batch_logits = batch_logits.detach().cpu()
        y_spt = y_spt.detach().cpu()

    return batch_logits, y_spt,loss
def get_repr_DTI_0shot(batch_data,tokenizer,chem_dict,protein_dict,args):
    #  . . . .  chemicals  . . . .
    chem_smiles = chem_dict[batch_data['InChIKey'].values.tolist()].values.tolist()
    chem_graph_list = []
    for smiles in chem_smiles:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_graph_data_obj_simple(mol)
        chem_graph_list.append(graph)
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=batch_data.shape[0],
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch
    #  . . . .  proteins  . . . .
    # if prot_descriptor_choice =='DISAE':
    uniprot_list = batch_data['uniprot+pfam'].values.tolist()
    protein_tokenized = torch.tensor([tokenizer.encode(protein_dict[uni]) for uni in uniprot_list  ])
    #  . . . .  to return  . . . .
    chem_graphs = chem_graphs.to(args['device'])
    protein_tokenized = protein_tokenized.to(args['device'])
    label = torch.LongTensor(list(batch_data['Activity'].values)).to(args['device'])
    return chem_graphs, protein_tokenized,label

def core_batch_prediction_0shot(batch_data_pertask, args, tokenizer,loss_fn,
                               chem_dict, protein_dict, model,detach=True):
    batch_chem_graphs_spt, batch_protein_tokenized_spt, y_spt = get_repr_DTI_0shot(batch_data_pertask,
                                                              tokenizer, chem_dict, protein_dict,
                                                              args)

    batch_logits = model(batch_protein_tokenized_spt, batch_chem_graphs_spt)
    loss = loss_fn(batch_logits,y_spt)
    if detach == True:
        batch_logits = batch_logits.detach().cpu()
        y_spt = y_spt.detach().cpu()
    return batch_logits, y_spt,loss


def get_repr_DTI(batch_data,tokenizer,chem_dict,protein_dict,prot_descriptor_choice):
    #  . . . .  chemicals  . . . .
    chem_smiles = chem_dict[batch_data['InChIKey'].values.tolist()].values.tolist()
    chem_graph_list = []
    for smiles in chem_smiles:
        mol = Chem.MolFromSmiles(smiles)
        graph = mol_to_graph_data_obj_simple(mol)
        chem_graph_list.append(graph)
    chem_graphs_loader = DataLoader(chem_graph_list, batch_size=batch_data.shape[0],
                                    shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch
    #  . . . .  proteins  . . . .
    if prot_descriptor_choice =='DISAE':
        uniprot_list = batch_data['uniprot+pfam'].values.tolist()
        protein_tokenized = torch.tensor([tokenizer.encode(protein_dict[uni]) for uni in uniprot_list  ])


    return chem_graphs, protein_tokenized



#------------------
#  read data
#------------------

def load_training_data(exp_path,debug_ratio):
    def load_data(exp_path,file,debug_ratio):
        dataset = pd.read_csv(exp_path +file)
        cut = int(dataset.shape[0] * debug_ratio)
        print(file[:-3] + ' size:', cut)
        return dataset.iloc[:cut,:]

    train = load_data(exp_path,'train.csv',debug_ratio)
    dev   = load_data(exp_path,'dev.csv',debug_ratio)
    test  = load_data(exp_path,'test.csv',debug_ratio)

    return train, dev, test

#------------------
#  evaluate
#------------------

def evaluate_binary_predictions(label, predprobs):
    probs = np.array(predprobs)
    predclass = np.argmax(probs, axis=1)
    # --------------------------by label---
    bothF1 = metrics.f1_score(label, predclass, average=None)
    bothprecision = metrics.precision_score(label, predclass, average=None)
    bothrecall = metrics.recall_score(label, predclass, average=None)
    class0 = [bothF1[0], bothprecision[0], bothrecall[0]]
    class1 = [bothF1[1], bothprecision[1], bothrecall[1]]
    # -------------------------overall---
    f1 = metrics.f1_score(label, predclass, average='weighted')
    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    prec, reca, thresholds = metrics.precision_recall_curve(label, probs[:, 1], pos_label=1)
    aupr = metrics.auc(reca, prec)

    overall = [f1,auc,aupr]
    return overall,class0,class1