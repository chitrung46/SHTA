import os
import torch
import random
import numpy as np
import argparse
import configparser
import logging
from datetime import datetime
from metrics import MAE_torch, RMSE_torch

def parse_args(device):
    args = argparse.ArgumentParser(prefix_chars='--', description='pretrain_arguments')
    args.add_argument('--dataset', default='PEMS08', type=str, required=True)
    args.add_argument('--mode', default='train', type=str, required=True)
    args.add_argument('--device', default=device, type=str, help='indices of GPUs')
    args.add_argument('--cuda', default=True, type=bool)
    args.add_argument('--model', default='SHTA', type=str)

    args_get, _ = args.parse_known_args()

    # get configuration
    config_file =  './config/{}.conf'.format(args_get.dataset)
    config = configparser.ConfigParser()
    config.read(config_file)
    # data
    args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('--lag', default=config['data']['lag'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    
    # model
    args.add_argument('--in_dim', default=config['model']['in_dim'], type=int)
    args.add_argument('--out_dim', default=config['model']['out_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--embed_dim_spa', default=config['model']['embed_dim_spa'], type=int)
    args.add_argument('--hidden_dim', default=config['model']['hidden_dim'], type=int)
    args.add_argument('--channels', type=int, default=config['model']['channels'])
    args.add_argument('--dropout', type=float, default=config['model']['dropout'])
    args.add_argument('--dynamic', type=str, default=config['model']['dynamic'])
    args.add_argument('--memory_size', type=int, default=config['model']['memory_size'])
    # train
    args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('--seed', default=config['train']['seed'], type=int)
    args.add_argument('--seed_mode', default=config['train']['seed_mode'], type=eval)
    args.add_argument('--tod', default=config['data']['tod'], type=eval)
    args.add_argument('--xavier', default=config['train']['xavier'], type=eval)
    args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('--epochs', default=config['train']['epochs'], type=int)
    args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('--debug', default=config['train']['debug'], type=eval)
    args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')
    args.add_argument('--load_pretrain_path', default=config['train']['load_pretrain_path'], type=str)
    args.add_argument('--save_pretrain_path', default=config['train']['save_pretrain_path'], type=str)
    args.add_argument('--change_epoch', default=config['train']['change_epoch'], type=int)
    args.add_argument('--up_epoch', default=config['train']['up_epoch'], type=str)
    
    # test
    args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=float)
    args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)

    # log
    args.add_argument('-log_dir', default='./', type=str)
    args.add_argument('-log_step', default=config['log']['log_step'], type=int)
    args.add_argument('-plot', default=config['log']['plot'], type=eval)

    args, _ = args.parse_known_args()
    args.filepath = './data/' + args.dataset +'/'
    args.filename = args.dataset
    A, Distance = get_adjacency_matrix(
            distance_df_filename=args.filepath + args.dataset + '.csv',
            num_of_vertices=args.num_nodes)
    args.A = A
    args.adj_mx = torch.FloatTensor(A)
    
    return args

def get_logger(root, name=None, debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger

def init_seed(seed, seed_mode):
    if seed_mode:
        torch.cuda.cudnn_enabled = False
        torch.backends.cudnn.deterministic = True
    # for quick running
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    if 'npy' in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        import csv
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)
        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))} 
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA
        else:
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    update_num = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print('Total params num: {}, Update params num: {}'.format(total_num, update_num))
    print('*****************Finish Parameter****************')

def scaler_mae_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inv_transform(preds)
            labels = scaler.inv_transform(labels)
        mae, mae_loss = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss

def scaler_rmse_loss(scaler, mask_value):
    def loss(preds, labels, mask=None):
        if scaler:
            preds = scaler.inv_transform(preds)
            labels = scaler.inv_transform(labels)
        mae, mae_loss = RMSE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae, mae_loss
    return loss

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inv_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean
