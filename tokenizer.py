import torch
import multiprocessing as mp
from functools import partial

BASE_CHAR = 383
class Tokenizer():
    def __init__(self, vocab_size, merges_file=None) -> None:
        assert vocab_size >= BASE_CHAR, f'make sure vocab size is at least {BASE_CHAR}'
        self.vocab_size = max(BASE_CHAR,vocab_size)
        self.merges = {}
        
        if merges_file:
            self.load_merges_from_file(merges_file)
        
    def load_merges_from_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        first, second, token = map(int, parts)
                        self.merges[(first, second)] = token
            print(f"Loaded {len(self.merges)} merges from {file_path}")
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
        except ValueError:
            print(f"Error: Invalid data format in {file_path}.")
    
    #Get frequency count for each pair of tokens in text
    def get_pair_counts(self, text):
        pairs = {}
        for i in range(len(text)-1):
            pair = text[i],text[i+1]
            pairs[pair] = pairs.get(pair,0)+1
        return pairs

    #Replace pairs in the text with a new token
    def merge_pair(self, text, merge_pair, new_token):
        new_text = []
        i=0
        while i < len(text):
            pair = text[i], -1 if i+1 >= len(text) else text[i+1]
            
            if pair == merge_pair:
                new_text.append(new_token)
                i+=2
            else:
                new_text.append(text[i])
                i+=1
        return new_text

    #Encode bytes into tokens using vocab of size up to vocab size.
    def encode(self, text, info=False):
        if type(text) == type(''):
            text = [c for c in text.encode('utf-8')]
        if type(text) == type(torch.tensor([])):
            text = text.tolist()
        max_token= max(BASE_CHAR,max(text))
        merges_number = self.vocab_size - max_token
        assert merges_number >= 0, 'Maximum token greater than vocab size'
        
        for index, (pair, token) in enumerate(self.merges.items()):
            text = self.merge_pair(text,pair, token)
            if info:
                print(f'Merged {index+1}/{len(self.merges)} {pair} -> {token}')
        
        while len(text) > 1 and len(self.merges) < merges_number:
            pairs = self.get_pair_counts(text)
            new_token = BASE_CHAR + len(self.merges)
            merge_pair = max(pairs,key=pairs.get)
            self.merges[merge_pair] = new_token
            print(f'{merge_pair}: {new_token}')
            text = self.merge_pair(text,merge_pair,new_token)
        return torch.tensor(text)

    #Decode tokens into bytes
    def decode(self, text):
        if type(text) == type(''):
            text = [c for c in text.encode('utf-8')]     
        if type(text) == type(torch.tensor([])):
            text = text.tolist()   
        for pair, token in reversed(self.merges.items()):
            i = 0
            new_text = []
            while i < len(text):
                if text[i] == token:
                    new_text.extend(pair)
                else:
                    new_text.append(text[i])
                i+=1
            text = new_text
        
        return bytes(new_text).decode('utf-8', errors='replace')#''.join(chr(c) for c  in text)#

    def parallel_encode(self, text, num_processes=4):
        if isinstance(text, str):
            text = [c for c in text.encode('utf-8')]
        if isinstance(text, torch.Tensor):
            text = text.tolist()

        chunk_size = len(text) // num_processes
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        with mp.Pool(processes=num_processes) as pool:
            encoded_chunks = pool.map(self.encode, chunks)

        return torch.cat(encoded_chunks)

    def parallel_decode(self, text, num_processes=4):
        if isinstance(text, str):
            text = [c for c in text.encode('utf-8')]
        if isinstance(text, torch.Tensor):
            text = text.tolist()

        chunk_size = len(text) // num_processes
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        with mp.Pool(processes=num_processes) as pool:
            decoded_chunks = pool.map(self.decode, chunks)

        return ''.join(decoded_chunks)