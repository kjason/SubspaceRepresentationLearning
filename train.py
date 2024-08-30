"""
Created on Mon Aug 14 2023

@author: Kuan-Lin Chen

Modified from https://github.com/kjason/DnnNormTimeFreq4DoA/tree/main/SpeechEnhancement
"""
import sys
import os
import time
import torch
import scipy.io
import math
from datetime import datetime
from utils import get_device_name

from batch_sampler import ConsistentRankBatchSampler

class TrainParam:
    def __init__(self,
                mu,
                mu_scale,
                mu_epoch,
                weight_decay,
                momentum,
                batch_size,
                val_batch_size,
                nesterov,
                onecycle,
                optimizer
                ):
        assert len(mu_scale)==len(mu_epoch), "the length of mu_scale and mu_epoch should be the same"        
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.max_epoch = mu_epoch[-1]
        self.mu = mu
        self.mu_scale = mu_scale
        self.mu_epoch = mu_epoch
        self.nesterov = nesterov
        self.onecycle = onecycle
        self.optimizer = optimizer

class TrainRegressor:
    pin_memory = True
    ckpt_filename = 'train.pt'
    def __init__(self,
                name,
                net,
                tp,
                trainset,
                validationset,
                criterion,
                device,
                seed,
                resume,
                checkpoint_folder,
                num_workers,
                consistent_rank_sampling,
                milestone = [],
                print_every_n_batch = 1,
                fp16 = False,
                meta_data = None
                ):
        torch.manual_seed(seed)
        self.criterion = criterion
        self.device = device
        self.net = net().to(device)
        self.checkpoint_folder = checkpoint_folder
        self.name = name
        self.seed = seed
        self.num_workers = num_workers
        self.milestone = milestone
        self.print_every_n_batch = print_every_n_batch
        self.consistent_rank_sampling = consistent_rank_sampling
        self.trainset = trainset
        self.validationset = validationset
        self.fp16 = fp16

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] {get_device_name(device)}")
        self.num_parameters = self.count_parameters()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] number of parameters in the model {name}: {self.num_parameters:,}")

        if self.consistent_rank_sampling is True:
            train_batch_sampler = ConsistentRankBatchSampler(N=trainset.N_datapoints_per_nsrc,K=len(trainset.num_sources),batch_size=tp.batch_size)
            val_batch_sampler = ConsistentRankBatchSampler(N=validationset.N_datapoints_per_nsrc,K=len(validationset.num_sources),batch_size=tp.batch_size)
            self.trainloader = torch.utils.data.DataLoader(trainset,batch_sampler=train_batch_sampler,num_workers=self.num_workers,pin_memory=self.pin_memory)
            self.validationloader = torch.utils.data.DataLoader(validationset,batch_sampler=val_batch_sampler,num_workers=self.num_workers,pin_memory=self.pin_memory)
        else:
            self.trainloader = torch.utils.data.DataLoader(trainset,batch_size=tp.batch_size,shuffle=True,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)
            self.validationloader = torch.utils.data.DataLoader(validationset,batch_size=tp.val_batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=self.pin_memory,drop_last=False)

        if tp.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=tp.mu,momentum=tp.momentum,nesterov=tp.nesterov,weight_decay=tp.weight_decay)
        elif tp.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.net.parameters(),lr=tp.mu,weight_decay=tp.weight_decay)
        else:
            raise ValueError(f"optimizer {self.tp.optimizer} not implemented")
        
        if tp.onecycle is True:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=tp.mu,steps_per_epoch=len(self.trainloader),epochs=tp.max_epoch)
        else:
            self.mu_lambda = lambda i: next(tp.mu_scale[j] for j in range(len(tp.mu_epoch)) if min(tp.mu_epoch[j]//(i+1),1.0) >= 1.0) if i<tp.max_epoch else 0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=self.mu_lambda)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.tp = tp
        self.total_train_time = 0
        self.start_epoch = 1
        self.train_loss = []
        self.validation_loss = []
        self.best_validation_loss = sys.float_info.max
        self.ckpt_path = os.path.join(self.checkpoint_folder,self.name,self.ckpt_filename)

        if resume is True and os.path.isfile(self.ckpt_path):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] resuming {self.name} from a checkpoint at {self.ckpt_path}",flush=True)
            self.__load()
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] ready to train {self.name} from scratch...",flush=True)
            init_validation_loss = self.validation()
            if self.consistent_rank_sampling is True:
                val_batch_sampler = ConsistentRankBatchSampler(N=self.validationset.N_datapoints_per_nsrc,K=len(self.validationset.num_sources),batch_size=self.tp.batch_size)
                self.validationloader = torch.utils.data.DataLoader(self.validationset,batch_sampler=val_batch_sampler,num_workers=self.num_workers,pin_memory=self.pin_memory)
            self.init_validation_loss = init_validation_loss
            self.best_validation_loss = init_validation_loss
            self.__save_net('init_model.pt')
            self.__save(0)
            self.__save_meta_data(meta_data)
 
    def __get_lr(self):
            for param_group in self.optimizer.param_groups:
                return param_group['lr']

    def __check_folder(self):
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        if not os.path.isdir(os.path.join(self.checkpoint_folder,self.name)):
            os.mkdir(os.path.join(self.checkpoint_folder,self.name))

    def __load(self):
        # Load checkpoint.
        checkpoint = torch.load(self.ckpt_path,map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.best_validation_loss = checkpoint['best_validation_loss']
        self.start_epoch = checkpoint['epoch']+1
        self.train_loss = checkpoint['train_loss']
        self.validation_loss = checkpoint['validation_loss']
        self.total_train_time = checkpoint['total_train_time']
        self.init_validation_loss = checkpoint['init_validation_loss']

    def __save_meta_data(self,meta_data):
        if meta_data is not None:
            self.__check_folder()
            torch.save(meta_data, os.path.join(self.checkpoint_folder,self.name,'meta_data.pt'))

    def __save_net(self,filename):
        self.__check_folder()
        net_path = os.path.join(self.checkpoint_folder,self.name,filename)
        torch.save(self.net.state_dict(), net_path)
        print('{} [train_regressor.py] model saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),net_path))

    def __save(self,epoch):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'init_validation_loss': self.init_validation_loss,
            'best_validation_loss': self.best_validation_loss,
            'epoch': epoch,
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'num_param': self.num_parameters,
            'seed': self.seed,
            'mu': self.tp.mu,
            'mu_scale': self.tp.mu_scale,
            'mu_epoch': self.tp.mu_epoch,
            'weight_decay': self.tp.weight_decay,
            'momentum': self.tp.momentum,
            'batch_size': self.tp.batch_size,
            'total_train_time': self.total_train_time,
            }
        self.__check_folder()
        torch.save(state, self.ckpt_path)
        print('{} [train_regressor.py] checkpoint saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),self.ckpt_path))
        del state['net'], state['optimizer'], state['scheduler'], state['scaler']
        state_path = os.path.join(self.checkpoint_folder,self.name,'train.mat')
        scipy.io.savemat(state_path,state)
        print('{} [train_regressor.py] state saved at {}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),state_path))
    
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def train(self):
        for i in range(self.start_epoch,self.tp.max_epoch+1):
            lr = self.__get_lr()
            num_batch = len(self.trainloader)
            tic = time.time()
            train_loss = self.__train(i)
            toc = time.time()
            self.total_train_time += (toc-tic)
            validation_loss = self.validation()
            print('{} [train_regressor.py] [Validation] epoch: {:4d}/{} batch: {:6d}/{} lr: {:.1e} loss: {:11.4e} best: {:11.4e} | training speed: {:.2f} seconds/epoch'.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                i,
                self.tp.max_epoch,
                num_batch,
                num_batch,
                lr,
                validation_loss,
                min(self.best_validation_loss,validation_loss),
                self.total_train_time/i
                ),flush=True)
 
            self.train_loss.append(train_loss)
            self.validation_loss.append(validation_loss)
            
            if validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                self.__save_net('best_model.pt')

            for k in self.milestone:
                if k==i:
                    self.__save_net('epoch_'+str(k)+'_model.pt')
                    self.__save(k)

            if math.isnan(train_loss):
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] NaN train loss... break the training loop".format())
                break
            if self.consistent_rank_sampling is True:
                train_batch_sampler = ConsistentRankBatchSampler(N=self.trainset.N_datapoints_per_nsrc,K=len(self.trainset.num_sources),batch_size=self.tp.batch_size)
                val_batch_sampler = ConsistentRankBatchSampler(N=self.validationset.N_datapoints_per_nsrc,K=len(self.validationset.num_sources),batch_size=self.tp.batch_size)
                self.trainloader = torch.utils.data.DataLoader(self.trainset,batch_sampler=train_batch_sampler,num_workers=self.num_workers,pin_memory=self.pin_memory)
                self.validationloader = torch.utils.data.DataLoader(self.validationset,batch_sampler=val_batch_sampler,num_workers=self.num_workers,pin_memory=self.pin_memory)
        if self.start_epoch<self.tp.max_epoch+1:
            self.__save_net('last_model.pt')
            self.__save(i)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] end of training at epoch {i} for the model saved at {self.ckpt_path}")
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] the model {self.ckpt_path} has already been trained for {self.tp.max_epoch} epochs")
        return self
    
    def __train(self,epoch_idx):
        tic = time.time()
        self.net.train()
        accumulated_train_loss = 0
        total = 0
        torch.manual_seed(self.seed+epoch_idx)
        lr = self.__get_lr()
        num_batch = len(self.trainloader)
        for batch_idx, (inputs, targets, source_numbers, angles) in enumerate(self.trainloader,1):
            inputs, targets, source_numbers, angles = inputs.to(self.device), targets.to(self.device), source_numbers.to(self.device), angles.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(enabled=self.fp16, device_type='cuda', dtype=torch.float16):
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, source_numbers, angles)
                batch_mean_loss = torch.mean(loss)

            if torch.isnan(batch_mean_loss):
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py] Nan train loss detected. The previous train loss: {train_loss:.6f}")
                return float("nan")

            self.scaler.scale(batch_mean_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            accumulated_train_loss += torch.sum(loss).item()

            total += loss.numel()

            train_loss = accumulated_train_loss/total
            toc = time.time()
            if (batch_idx-1)%self.print_every_n_batch == 0 or batch_idx == num_batch:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [train_regressor.py]      [Train] "
                      f"epoch: {epoch_idx:4d}/{self.tp.max_epoch} batch: {batch_idx:6d}/{num_batch} lr: {lr:.1e} "
                      f"loss: {train_loss:11.4e} | ELA: {self.total_train_time+toc-tic:.3e}s",flush=True)
            if self.tp.onecycle is True:
                self.scheduler.step()
        if self.tp.onecycle is False:
            self.scheduler.step()
        return train_loss
    
    def validation(self):
        self.net.eval()
        accumulated_validation_loss = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets, source_numbers, angles) in enumerate(self.validationloader,1):
                inputs, targets, source_numbers, angles = inputs.to(self.device), targets.to(self.device), source_numbers.to(self.device), angles.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, source_numbers, angles)
                
                accumulated_validation_loss += torch.sum(loss).item()
                total += loss.numel()
                
        validation_loss = accumulated_validation_loss/total
        return validation_loss