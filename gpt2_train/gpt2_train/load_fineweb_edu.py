# %%
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm
import random

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"

shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# %%


idx = list(range(len(fw)))
random.shuffle(idx)
idx[:5]

# print (len(fw))
# print(fw[1])
# for i in fw:
#     print(i['text'])
#     break
