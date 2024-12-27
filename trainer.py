# trainer.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as nn_utils
from dataclasses import dataclass
import time
import math
import wandb
import os
from datetime import datetime

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, i):
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + self.block_size + 1]
        return x, y

@dataclass
class TrainConfig:
    max_iters: int = 5000
    eval_interval: int = 200
    eval_iters: int = 50
    total_batch_size: int = 500000  # in tokens
    batch_size: int = 16
    learning_rate: float = 5e-4
    min_lr: float = 5e-7
    num_workers: int = 0
    const_iter_ratio: float = 0.95
    min_lr_decay_ratio: float = 0.1
    min_lr_iter_ratio: float = 0.1
    warmup_iter_ratio: float = 0.05
    grad_clip: float = 1.0
    checkpoint_interval: int = 1000  # Save a checkpoint every 1000 iterations
    checkpoint_dir: str = "checkpoints"  # Directory to save checkpoints

class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, config=TrainConfig):
        self.config = config
        self.train_dataloader = DataLoader(
            train_dataset,
            self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        self.val_dataloader = (
            DataLoader(
                val_dataset,
                config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
            if val_dataset is not None
            else None
        )
        self.model = model
        self.block_size = model.config.block_size
        self.device = model.device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0,
        )
        self.grad_accum_steps = (
            self.config.total_batch_size // (self.config.batch_size * self.block_size)
        )
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        loader_dict = {"train": self.train_dataloader}
        if self.val_dataloader is not None:
            loader_dict["val"] = self.val_dataloader

        for split, dataloader in loader_dict.items():
            losses = torch.zeros(self.config.eval_iters)
            data_iter = iter(dataloader)
            for k in range(self.config.eval_iters):
                X, Y = next(data_iter)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(X.to(self.device), Y.to(self.device))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def get_lr(self, cur_iter):
        config = self.config
        warmup_end_iter = config.warmup_iter_ratio * config.max_iters
        if cur_iter < warmup_end_iter:
            return (
                config.min_lr
                + (config.learning_rate - config.min_lr) * (cur_iter / warmup_end_iter)
            )
        const_lr_iter = config.const_iter_ratio * config.max_iters
        if cur_iter < const_lr_iter:
            return config.learning_rate

        min_lr_decayed = config.min_lr_decay_ratio * config.learning_rate
        if cur_iter < config.max_iters:
            return (
                min_lr_decayed
                + math.cos(
                    (cur_iter - const_lr_iter)
                    / (config.max_iters - const_lr_iter)
                    * math.pi
                    / 2
                )
                * (config.learning_rate - min_lr_decayed)
            )
        return min_lr_decayed

    def save_checkpoint(self, iteration):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"model_checkpoint_{iteration}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        wandb.save(checkpoint_path) # Save to wandb too

    def train(self):
        self.model.train()
        config = self.config
        data_iter = iter(self.train_dataloader)
        for i in range(config.max_iters):
            lr = self.get_lr(i)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if config.eval_iters != 0 and (
                i % config.eval_interval == 0 or i == config.max_iters - 1
            ):
                losses = self.estimate_loss()
                print(f"step {i} losses: {losses}")
                # Log losses to W&B
                wandb.log({"train_loss": losses["train"], "iter": i})
                if "val" in losses:
                    wandb.log({"val_loss": losses["val"], "iter": i})

            if i > 0 and i % config.checkpoint_interval == 0:
                self.save_checkpoint(i)

            t0 = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)

            iter_loss = 0.0
            for _ in range(self.grad_accum_steps):
                try:
                    xb, yb = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    xb, yb = next(data_iter)

                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(xb.to(self.device), yb.to(self.device))
                iter_loss += loss.detach()
                loss.backward()

            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=config.grad_clip)
            self.optimizer.step()

            iter_loss /= self.grad_accum_steps
            t1 = time.perf_counter()
            dt = t1 - t0

            # Log metrics to W&B
            wandb.log(
                {
                    "loss": iter_loss,
                    "lr": lr,
                    "time": dt,
                    "iter": i,
                }
            )

            print(
                f"step {i} | loss: {iter_loss:.4f} | time: {dt*1000:.2f} ms | lr: {lr:.7f}"
            )

        # Save one last checkpoint at the end of training
        self.save_checkpoint(config.max_iters)