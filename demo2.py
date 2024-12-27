# %%
from model import WieszczGPT, GPTConfig
from trainer import Trainer, TextDataset, TrainConfig
from datasets import load_dataset
import multiprocessing as mp
import torch
import tiktoken
import numpy as np
import random
import os
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
os.makedirs('data', exist_ok=True)
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

tokenizer = tiktoken.get_encoding("gpt2")

def encode_shard(shard):
    # print(shard)
    if len(shard["text"])<1: return []
    # print(shard["text"])
    # return np.arange(random.randint(10,10))
    encoded = tokenizer.encode(shard["text"],allowed_special="all")
    return encoded
    # return np.array([ord(c) for c in shard],dtype=np.int32)
def encode_dataset(dataset, shard_size, num_workers=2):
    with mp.Pool(num_workers) as pool:
        shard_index = 0
        shard_tokens = np.zeros(shard_size, dtype=np.int32)
        current_index = 0
        prev_logged_index = 0
        
        for tokens in pool.imap(encode_shard, dataset, chunksize=2**13):
            # tokens = encode_shard(document)  # Assuming encode_shard tokenizes a document
    
            if len(tokens) <= shard_size - current_index:
                shard_tokens[current_index:current_index + len(tokens)] = tokens
                current_index += len(tokens)
            else:
                tokens_fit = shard_size - current_index
                tokens_overflowed = len(tokens) - tokens_fit
    
                while tokens_overflowed >= 0:
                    
                    tokens_fit = min(shard_size - current_index, len(tokens)) # take the minimum between available space and tokens available
                    
                    shard_tokens[current_index:] = tokens[:tokens_fit]
                    print(f'data/shard{shard_index}.bin')
    
                    np.save(f'data/shard{shard_index}.bin', shard_tokens) # Uncomment to save to disk
    
                    tokens = tokens[tokens_fit:]
                    current_index = 0 # current_index is reset to 0, since we saved the shard
                    prev_logged_index = 0
                    tokens_overflowed -= shard_size
                    shard_index += 1
            if current_index-prev_logged_index > 0.01*shard_size:
                print(f"{current_index}/{shard_size}")
                prev_logged_index = current_index
        
        # if current index is not 0, it means that we still have some token left
        if current_index > 0:
            np.save(f'data/shard{shard_index}.bin', shard_tokens[:current_index])
        
    return encoded
# %%
if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() - 2)
    print(f"Begin tokenization with {nprocs} workers")
    
    encoded = encode_dataset(dataset,10**8,nprocs)