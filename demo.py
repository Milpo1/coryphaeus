# init.py
import torch
import numpy as np
import os
import mmap
import tiktoken
from dataclasses import dataclass
from datetime import datetime

# Assuming your custom modules are defined here or imported
from model import WieszczGPT, GPTConfig
from trainer import Trainer, TextDataset, TrainConfig

import wandb  # Import Weights & Biases

# Initialize Weights & Biases

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def read_encoded_file_to_tensor(file_path):
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            np_array = np.frombuffer(mmapped_file, dtype=np.int32)
            tensor = torch.from_numpy(np.array(np_array))
        print(f"Successfully mapped {len(tensor)} tokens from {file_path}")
        return tensor
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except ValueError as e:
        print(f"Error: Invalid data format in {file_path}. {str(e)}")
        return None

# Model configuration
model_config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_embed=768,
    n_heads=12,
    n_layers=12,
    dropout=0.05,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Data loading
data = read_encoded_file_to_tensor('data/shard0.bin.npy')
data = data[32:]
if data is not None:
    print(f"Tensor shape: {data.shape}")
    print(f"First few elements: {data[:100]}")

# Data splitting
n = int(0.8 * len(data))
train_dataset = TextDataset(data[:n], model_config.block_size)
val_dataset = TextDataset(data[n:], model_config.block_size)

# Model initialization
torch.set_float32_matmul_precision('high')
model = WieszczGPT(model_config)
state_dict = torch.load('model_state_dict_2024-12-27_18-30-06.pth', map_location=device)
model.to(device)

# Remove '_orig_mod.' prefix from keys, if present
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

# Load state_dict
missing, unexpected = model.load_state_dict(state_dict)
if missing or unexpected:
    print('Something went from when loading model state dict. missing, unexpected:')
    print(missing)
    print(unexpected)
print('Model state dict loaded successfully')
if torch.cuda.get_device_capability(0)[0] >= 7:
    model = torch.compile(model)

# Training configuration
train_config = TrainConfig(
    max_iters=3000,
    eval_iters=250,
    eval_interval=1000,
    total_batch_size=model_config.block_size * 36 * 14,
    batch_size=36,
    checkpoint_interval=500 # Save checkpoint every 500 iterations
)
wandb.init(project="wieszcz-gpt",
            config={
                'max_iters':3000,
                'eval_iters':250,
                'eval_interval':1000,
                'total_batch_size':model_config.block_size * 36 * 14,
                'batch_size':36,
                'checkpoint_interval': 500
              }
          )

# Log model graph to W&B
wandb.watch(model)

# Trainer initialization
trainer = Trainer(model, train_dataset, val_dataset, train_config)

# Training
trainer.train()

# Get the current datetime
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the tokenizer with datetime in the filename
torch.save(model.state_dict(),f"model_state_dict_{current_datetime}.pth")

# Generation (after training)
model.eval()

def gen(text, max_new_tokens = 1,temperature=1):
    in_tokens = tokenizer.encode(text)
    context_tensor = torch.zeros(model_config.block_size, dtype=torch.long)
    context_tensor[-len(in_tokens):]= torch.tensor(in_tokens, dtype=torch.long)
    context_tensor = context_tensor.reshape(1,-1)
    context_tensor = context_tensor.to(device)
    # temperature = torch.tensor(temperature, dtype=torch.float).to(device)
    # max_new_tokens = torch.tensor(max_new_tokens, dtype=torch.int32).to(device)
    gen = model.generate(context_tensor, max_new_tokens=max_new_tokens,temperature=temperature)[0].tolist()
    return text+tokenizer.decode(gen[-max_new_tokens:])

print(gen("Hello",100,1))
# Finish W&B run
wandb.finish()