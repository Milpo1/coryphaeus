# %%
from tokenizer import Tokenizer
from model import WieszczGPT, GPTConfig
from trainer import Trainer, TextDataset, TrainConfig
import torch
import mmap 
import numpy as np
import os
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("gpt2")
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
set_seed(42)

# import os

# def load_all_text_files_into_string(directory_path):
#     all_text = []
#     for root, _, files in os.walk(directory_path):
#         for file in files:
#             if file.endswith(".txt"):
#                 file_path = os.path.join(root, file)
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     all_text.append(f.read())
#     return "\n".join(all_text)

# def save_text_to_file(text, output_file_path):
#     with open(output_file_path, 'w', encoding='utf-8') as f:
#         f.write(text)

# # Replace 'your_directory_path' with the actual directory path
# directory_path = 'plwiki3'
# # Replace 'output_file.txt' with the desired output file path
# output_file_path = 'wiki.txt'

# all_text_content = load_all_text_files_into_string(directory_path)
# save_text_to_file(all_text_content, output_file_path)

# print(f"All text has been concatenated and saved to {output_file_path}")
def read_encoded_file_to_tensor(file_path):
    try:
        # Get the file size
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'rb') as file:
            # Memory-map the file
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Interpret the memory-mapped file as a NumPy array
            # Assuming the C++ code saved integers as 4-byte (32-bit) values
            np_array = np.frombuffer(mmapped_file, dtype=np.int16)
            
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(np.array(np_array))
        
        print(f"Successfully mapped {len(tensor)} tokens from {file_path}")
        return tensor
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except ValueError as e:
        print(f"Error: Invalid data format in {file_path}. {str(e)}")
        return None

# with open('wiki_tiny.txt', 'r') as file:
#     token_train_data = file.read()
# with open('wiki.txt', 'r') as file:
#     raw_data = file.read()
    
# %%
model_config = GPTConfig(block_size=384,vocab_size=500,n_embed=512,n_heads=8,n_layers=4,dropout=0.15)

tok = Tokenizer(model_config.vocab_size,'merges.txt')
# tok.encode(raw_data,True)
# del token_train_data
#data = tok.encode(raw_data)
#del raw_data
# %%
# data = tokenizer.encode(token_train_data)
data = read_encoded_file_to_tensor('output.bin')

if data is not None:
    print(f"Tensor shape: {data.shape}")
    print(f"First few elements: {data[:10]}")
# %%
n = int(0.8*len(data))
train_dataset = TextDataset(data[:n],model_config.block_size)
val_dataset = TextDataset(data[n:],model_config.block_size)

model = WieszczGPT(model_config)
# model.device = 'cpu'
model.to(model.device)
if torch.cuda.get_device_capability(0)[0] >= 7:
    model = torch.compile(model)

train_config = TrainConfig(max_iters=4000,eval_iters=100,eval_interval=1000,total_batch_size=384*32,batch_size=32)


trainer = Trainer(model,train_dataset, val_dataset,train_config)
# %%
trainer.train()
# %%
model.eval()
context = torch.zeros((1, model_config.block_size), dtype=torch.long, device=model.device)
#context = tok.encode('Litwo! Ojczyzno moja! ty jesteś jak zdrowie:\nIle cię trzeba cenić, ten tylko się dowie,\nKto cię stracił. Dziś piękność twą w całej ozdobie\nWidzę i opisuję, bo tęsknię po tobie.Panno święta, co Jasnej bronisz Częstochowy\nI w Ostrej świecisz Bramie! Ty, co gród zamkowy\nNowogródzki ochraniasz z jego wiernym ludem!')
#context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(dim=0)
gen = model.generate(context, max_new_tokens=1000)[0].tolist()
print(tok.decode(gen))



# %%
torch.save(model,'model.pth')

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Set up pre-tokenization and normalization
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Set up a trainer
trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=3,limit_alphabet=300)

# Train the tokenizer on your dataset
files = ["D:\proj\wieszcz-gpt3\wiki_cleaned.txt"]

tokenizer.train(files, trainer)
# %%
# Save the tokenizer
tokenizer.save("my_custom_tokenizer.json")

with open('wiki_tiny.txt', 'r') as file:
    text = file.read()
    
tokenizer.encode(text)

# %%
import re

# Read the contents of the file
with open('wiki.txt', 'r') as file:
    text = file.read()

# Use regex to remove text enclosed in <>
cleaned_text = re.sub(r'<[^>]*>', '<END>', text)

# Optionally, write the cleaned text back to a new file
with open('wiki_cleaned.txt', 'w') as file:
    file.write(cleaned_text)

print("Text cleaning complete. Check 'output.txt' for the result.")
