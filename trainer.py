import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as nn_utils
from dataclasses import dataclass
import time
import math

shard_size = 10e5


class TextDataset(Dataset):
    def __init__(self, data, block_size) -> None:
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data)-self.block_size-1
    def __getitem__(self, i):
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+self.block_size+1]      
        return x, y

@dataclass
class TrainConfig:
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 50
    total_batch_size: int = 500000 # in tokens
    batch_size: int = 16
    learning_rate: float = 5e-4
    min_lr: float = 5e-7
    num_workers: int = 0
    min_lr_decay_ratio: float = 0.1
    min_lr_iter_ratio: float = 0.1
    warmup_iter_ratio: float = 0.1
    grad_clip: float = 1
class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, config=TrainConfig):
        assert train_dataset.__len__() > 0
        self.config = config
        self.train_dataloader = DataLoader(train_dataset,self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
                                    # sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)))
        self.val_dataloader = DataLoader(val_dataset,config.batch_size, shuffle=False, num_workers=self.config.num_workers)\
                                        if val_dataset is not None else None
                                    # sampler=torch.utils.data.RandomSampler(val_dataset, replacement=True, num_samples=int(1e10))) \
        self.model = model
        self.block_size = model.config.block_size
        self.device = model.device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,
                                            betas=(0.9,0.95),eps=1e-8,weight_decay=0)#  TODO weight decay
        assert self.config.total_batch_size % (self.config.batch_size*self.block_size) == 0, 'total_batch_size should be divisible by batch_size*block_size'
        self.grad_accum_steps = self.config.total_batch_size // (self.config.batch_size*self.block_size)
        # torch.set_float32_matmul_precision('high')
        
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        config = self.config
        self.model.eval()
        loader_dict = {'train':self.train_dataloader}
        if self.val_dataloader is not None:
            loader_dict['val'] = self.val_dataloader
        
        for split, dataloader in loader_dict.items():
            losses = torch.zeros(config.eval_iters)
            data_iter = iter(dataloader)
            for k in range(config.eval_iters):
                X, Y = next(data_iter)
                # with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = self.model(X.to(self.device), Y.to(self.device))
                losses[k] = loss.item()
            out[split] = str(losses.mean().item())[:5]
        self.model.train()
        return out
    
    def get_lr(self, cur_iter):
        """
        LR ^
        |
        |      /‾‾\          learning_rate
        |     /    ‾‾\
        |    /        ‾\
        |   /           ‾\
        |  /              \  min_lr_decayed
        | /
        min_lr
        +------+----------+--> iter
        0     warmup_    max_
            end_iter   iters
        """
        #first  LR warmup  from min_lr to learning_rate at (warmup_iter_ratio % of iters)
        config = self.config
        warmup_end_iter = config.warmup_iter_ratio * config.max_iters
        if cur_iter < warmup_end_iter:
            return config.min_lr \
                + (config.learning_rate-config.min_lr) \
                * (cur_iter/warmup_end_iter)
        # cosine LR decay from learning_rate at (warmup_iter_ratio % of iters) down to min_lr_ratio at (min_lr_iter_ratio % of iters)
        min_lr_decayed = config.min_lr_decay_ratio * config.learning_rate
        if cur_iter < config.max_iters:
            return min_lr_decayed \
                    + math.cos((cur_iter - warmup_end_iter) / (config.max_iters - warmup_end_iter) * math.pi/2) \
                    * (config.learning_rate - min_lr_decayed)
        return min_lr_decayed

    def train(self):
        self.model.train()
        config = self.config
        data_iter = iter(self.train_dataloader)
        for i in range(config.max_iters): # one iter does one whole batch of size total_batch_size
            if i % config.eval_interval == 0 or i == config.max_iters - 1:
                #print(f'{i} estimating loss')
                
                losses = self.estimate_loss()
                print(f"step {i} losses: {losses}")
                
            t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            
            iter_loss = 0.0
            for _ in range(self.grad_accum_steps):    
                try:
                    xb, yb = next(data_iter)
                except StopIteration:
                    # Reset the iterator when it's exhausted
                    data_iter = iter(self.train_dataloader)
                    xb, yb = next(data_iter)
                    
                # with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits, loss = self.model(xb.to(self.device), yb.to(self.device))
                iter_loss += loss.detach()
                loss.backward()
            
            
            
            lr = self.get_lr(i)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=config.grad_clip)
            self.optimizer.step()
            
            iter_loss /= self.grad_accum_steps # average the loss over micro-batches
            t1 = time.perf_counter()
            dt = t1-t0
            print(f"step {i} loss: {iter_loss:.4f} time: {dt*1000:.2f} ms")
    