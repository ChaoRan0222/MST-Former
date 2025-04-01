import time
import random
import argparse
import numpy as np
import configparser
from tqdm import tqdm
from lib.metrics import _compute_loss
import math
import torch

# 从重构后的模型库中导入
from models.mst_former import MST_Former

from lib.data_loader import log_string
from lib.data_loader import loadData
from lib.metrics import metric

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        log_string(log, '\n------------ Loading Data -------------')
        self.trainX, self.trainY, self.trainXTE, self.trainYTE,\
        self.valX, self.valY, self.valXTE, self.valYTE,\
        self.testX, self.testY, self.testXTE, self.testYTE,\
        self.mean, self.std,\
        self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx = loadData(
                                        self.traffic_file, self.meta_file,
                                        self.input_len, self.output_len,
                                        self.train_ratio, self.test_ratio,
                                        self.adj_file, self.recur_times,
                                        self.tod, self.dow,
                                        self.spa_patchsize, log)
        log_string(log, '------------ End -------------\n')

        self.best_epoch = 0

        self.device = torch.device(f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        self.build_model()
    
    def build_model(self):
        # 参数映射与重命名
        temp_patch_size = self.tem_patchsize
        temp_patch_num = self.tem_patchnum
        attn_reduce_factor = self.factors
        partition_recur_depth = self.recur_times
        spatial_patch_size = self.spa_patchsize
        spatial_patch_num = self.spa_patchnum
        input_dim = self.input_dims
        node_embed_dim = self.node_dims
        tod_embed_dim = self.tod_dims
        dow_embed_dim = self.dow_dims
        
        self.model = MST_Former(temp_patch_size, temp_patch_num,
                            self.node_num, spatial_patch_size, spatial_patch_num,
                            self.tod, self.dow,
                            self.layers, attn_reduce_factor,
                            input_dim, node_embed_dim, tod_embed_dim, dow_embed_dim,
                            self.ori_parts_idx, self.reo_parts_idx, self.reo_all_idx).to(self.device)

        # 计算并打印模型总参数
        total_params = sum(p.numel() for p in self.model.parameters())
        log_string(log, f'Total number of parameters: {total_params:,}')

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate,weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[1,35,40],
            gamma=0.5,
        )

    def vali(self):
        self.model.eval()
        num_val = self.valX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.valX[start_idx : end_idx]
                    Y = self.valY[start_idx : end_idx]
                    TE = torch.from_numpy(self.valXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)

                    y_hat = self.model(NormX,TE)

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)

    def train(self):
        log_string(log, "======================TRAIN MODE======================")
        min_loss = 10000000.0
        num_train = self.trainX.shape[0]
        
        # 记录训练开始时间
        train_start_time = time.time()
        
        for epoch in tqdm(range(1,self.max_epoch+1)):
            # 记录每个epoch的开始时间
            epoch_start_time = time.time()
            
            self.model.train()
            train_l_sum, train_acc_sum, batch_count = 0.0, 0.0, 0
            permutation = np.random.permutation(num_train)
            self.trainX = self.trainX[permutation]
            self.trainY = self.trainY[permutation]
            self.trainXTE = self.trainXTE[permutation]
            num_batch = math.ceil(num_train / self.batch_size)
            with tqdm(total=num_batch) as pbar:
                for batch_idx in range(num_batch):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_train, (batch_idx + 1) * self.batch_size)

                    X = self.trainX[start_idx : end_idx]
                    Y = self.trainY[start_idx : end_idx]
                    TE = torch.from_numpy(self.trainXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)
                    
                    Y = torch.from_numpy(Y).float().to(self.device)
                    
                    self.optimizer.zero_grad()

                    y_hat = self.model(NormX,TE)

                    loss = _compute_loss(Y, y_hat*self.std+self.mean)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    self.optimizer.step()
                    
                    train_l_sum += loss.cpu().item()

                    batch_count += 1
                    pbar.update(1)
            
            # 计算当前epoch耗时
            epoch_time = time.time() - epoch_start_time
            
            log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.2f min'
                % (epoch, self.optimizer.param_groups[0]['lr'], 
                   train_l_sum / batch_count, epoch_time/60))
            
            mae, rmse, mape = self.vali()
            self.lr_scheduler.step()
            if mae[-1] < min_loss:
                self.best_epoch = epoch
                min_loss = mae[-1]
                torch.save(self.model.state_dict(), self.model_file)
        
        # 计算总训练时长
        total_train_time = time.time() - train_start_time
        hours = total_train_time // 3600
        minutes = (total_train_time % 3600) // 60
        seconds = total_train_time % 60
        
        log_string(log, f'Best epoch is: {self.best_epoch}')
        log_string(log, f'Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s')

    def test(self):
        log_string(log, "======================TEST MODE======================")
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        num_val = self.testX.shape[0]
        pred = []
        label = []

        num_batch = math.ceil(num_val / self.batch_size)
        with torch.no_grad():
            for batch_idx in range(num_batch):
                if isinstance(self.model, torch.nn.Module):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(num_val, (batch_idx + 1) * self.batch_size)

                    X = self.testX[start_idx : end_idx]
                    Y = self.testY[start_idx : end_idx]
                    TE = torch.from_numpy(self.testXTE[start_idx : end_idx]).to(self.device)
                    NormX = torch.from_numpy((X-self.mean)/self.std).float().to(self.device)

                    y_hat = self.model(NormX,TE)

                    pred.append(y_hat.cpu().numpy()*self.std+self.mean)
                    label.append(Y)
        
        pred = np.concatenate(pred, axis = 0)
        label = np.concatenate(label, axis = 0)

        maes = []
        rmses = []
        mapes = []

        for i in range(pred.shape[1]):
            mae, rmse , mape = metric(pred[:,i,:], label[:,i,:])
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            log_string(log,'step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, mae, rmse, mape))
        
        mae, rmse, mape = metric(pred, label)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, 'average, mae: %.4f, rmse: %.4f, mape: %.4f' % (mae, rmse, mape))
        
        return np.stack(maes, 0), np.stack(rmses, 0), np.stack(mapes, 0)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='configuration file')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type = int, default = config['train']['seed'])
    parser.add_argument('--batch_size', type = int, default = config['train']['batch_size'])
    parser.add_argument('--max_epoch', type = int, default = config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default = config['train']['learning_rate'])
    parser.add_argument('--weight_decay', type=float, default = config['train']['weight_decay'])

    parser.add_argument('--input_len', type = int, default = config['data']['input_len'])
    parser.add_argument('--output_len', type = int, default = config['data']['output_len'])
    parser.add_argument('--train_ratio', type = float, default = config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type = float, default = config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type = float, default = config['data']['test_ratio'])

    parser.add_argument('--layers', type=int, default = config['param']['layers'])
    parser.add_argument('--tem_patchsize', type = int, default = config['param']['temp_patch_size'])
    parser.add_argument('--tem_patchnum', type = int, default = config['param']['temp_patch_num'])
    parser.add_argument('--factors', type=int, default = config['param']['attn_reduce_factor'])
    parser.add_argument('--recur_times', type = int, default = config['param']['partition_recur_depth'])
    parser.add_argument('--spa_patchsize', type = int, default = config['param']['spatial_patch_size'])
    parser.add_argument('--spa_patchnum', type = int, default = config['param']['spatial_patch_num'])
    parser.add_argument('--node_num', type = int, default = config['param']['nodes'])
    parser.add_argument('--tod', type=int, default = config['param']['tod'])
    parser.add_argument('--dow', type=int, default = config['param']['dow'])
    parser.add_argument('--input_dims', type=int, default = config['param']['input_dim'])
    parser.add_argument('--node_dims', type=int, default = config['param']['node_embed_dim'])
    parser.add_argument('--tod_dims', type=int, default = config['param']['tod_embed_dim'])
    parser.add_argument('--dow_dims', type=int, default = config['param']['dow_embed_dim'])

    parser.add_argument('--traffic_file', default = config['file']['traffic'])
    parser.add_argument('--meta_file', default = config['file']['meta'])
    parser.add_argument('--adj_file', default = config['file']['adj'])
    parser.add_argument('--model_file', default = config['file']['model'])
    parser.add_argument('--log_file', default = config['file']['log'])

    args = parser.parse_args()

    log = open(args.log_file, 'w')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    log_string(log, '------------ Options -------------')
    for k, v in vars(args).items():
        log_string(log, '%s: %s' % (str(k), str(v)))
    log_string(log, '-------------- End ----------------')

    solver = Solver(vars(args))

    solver.train()
    solver.test()
    
