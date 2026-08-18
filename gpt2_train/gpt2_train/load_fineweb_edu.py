# %%
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
import random
from pathlib import Path

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"

shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
# %%
if "__file__" in locals():
    output_dir = Path(__file__).parent.parent/'dataset'
else:
    # in jupyter notebook
    output_dir = Path(os.getcwd()) / 'gpt2_train'/'dataset'

# %%


random.seed(42)
idx = list(range(len(fw)))
random.shuffle(idx)
idx[:5]

# %%
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc_idx: int):
    text = fw[doc_idx]['text']
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def main():
    with mp.Pool() as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        def write_file(data: np.ndarray):
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(output_dir, f"edufineweb_{split}_{shard_index:06d}")
            print(f"Saving {len(data)} tokens to {filename}")
            np.save(filename, data)

        for tokens in tqdm(pool.imap(tokenize, idx, chunksize=16)):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
            else:
                assert len(tokens) < shard_size
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_file(all_tokens_np)
                token_count = 0
                shard_index += 1
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder
        if token_count != 0:
            write_file(all_tokens_np[:token_count])

if __name__ == "__main__":
    main()


# print (len(fw))
# print(fw[1])
# for i in fw:
#     print(i['text'])
#     break
