# %%
from model import WieszczGPT, GPTConfig
from trainer import Trainer, TextDataset, TrainConfig
from datasets import load_dataset
import multiprocessing as mp
import torch
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

dataset = load_dataset("Salesforce/wikitext", 'wikitext-2-v1',split='train')

model_config = GPTConfig(block_size=384,vocab_size=5000,n_embed=256,n_heads=8,n_layers=4,dropout=0.15)

# %%
from tokenizers import Tokenizer
from tokenizers.models import BPE
filename = f"tokenizer-wiki-{model_config.vocab_size}.json"
if os.path.isfile(filename):
    tokenizer = Tokenizer.from_file(filename)
else:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    from tokenizers.trainers import BpeTrainer
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],vocab=model_config.vocab_size,limit_alphabet=256)

    # from tokenizers.pre_tokenizers import Whitespace
    # tokenizer.pre_tokenizer = Whitespace()

    # all_texts_iter = (row for split in ['train','validation','test'] for row in dataset[split]['text'])
    texts_iter = (row for row in dataset['text'])
    tokenizer.train_from_iterator(texts_iter, trainer)

    tokenizer.save(filename)

# %%
# def tokenize_example(examples):
#     # encode_batch returns a list of Encoding objects
#     encoded = tokenizer.encode_batch(examples["text"])
#     # Extract the `ids` (token IDs) from each Encoding object
#     input_ids = [enc.ids for enc in encoded]
#     

# # Try using a smaller batch_size and specify writer_batch_size
# tokenized_dataset = dataset['train'].map(
#     tokenize_example,
#     batched=True,
#     batch_size=100,          # adjust as needed
#     writer_batch_size=100    # adjust as needed
# )

def _load_data_shard(filename):
    pass

def _save_data_shard(filename):
    pass

def encode_shard(shard):
    # print(shard)
    if len(shard["text"])<1: return []
    # print(shard["text"])
    # return np.arange(random.randint(10,10))
    encoded = tokenizer.encode(shard["text"])
    return encoded.ids
    # return np.array([ord(c) for c in shard],dtype=np.int32)
def encode_dataset(dataset, shard_size, num_workers=2):
    shard_index = 0
    shard_tokens = np.zeros(shard_size, dtype=np.int32)
    current_index = 0
    encoded = []  # List to store the encoded shards

    for document in dataset:
        tokens = encode_shard(document)  # Assuming encode_shard tokenizes a document

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

                encoded.append(shard_tokens.copy().tolist())
                # np.save(f'data/shard{shard_index}.bin', shard_tokens) # Uncomment to save to disk

                tokens = tokens[tokens_fit:]
                current_index = 0 # current_index is reset to 0, since we saved the shard
                tokens_overflowed -= shard_size
                shard_index += 1
    
    # if current index is not 0, it means that we still have some token left
    if current_index > 0:
        encoded.append(shard_tokens[:current_index].copy().tolist())

    return encoded
# %%
if __name__ == '__main__':
    encoded = encode_dataset(dataset,3000,4)
    # print(''.join([chr(c) for c in encoded[-2]]))
#     encode_dataset(dataset[:10000],100,4)
# %%
