import pandas as pd
import numpy as np
import pickle5 as pickle
import json
import os
from datetime import datetime



##### JSON modules #####
def save_json(data,filename):
  with open(filename, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=4)

def load_json(filename):
  with open(filename, 'r') as fp:
    data = json.load(fp)
  return data

##### pickle modules #####
def save_dict_pickle(data,filename):
  with open(filename,'wb') as handle:
    pickle.dump(data,handle, pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
  with open(path, 'rb') as f:
    dict = pickle.load(f)
  return  dict

#####  #####
def set_up_exp_folder(cwd,exp = 'experiment_logs/'):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = cwd + exp
    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
    if os.path.exists(checkpoint_dir ) == False:
            os.mkdir(checkpoint_dir )
    return checkpoint_dir

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def set_up_exp_folder(cwd,exp = 'experiment_logs/'):
#     now = datetime.now()
#     timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
#     print('timestamp: ',timestamp)
#     save_folder = cwd + exp
#     if os.path.exists(save_folder) == False:
#             os.mkdir(save_folder)
#     checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
#     if os.path.exists(checkpoint_dir ) == False:
#             os.mkdir(checkpoint_dir )
#     return checkpoint_dir


