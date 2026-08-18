import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from .dataset_helpers import get_hellaswag_dir


output_dir = get_hellaswag_dir()
