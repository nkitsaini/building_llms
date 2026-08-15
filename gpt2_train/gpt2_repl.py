# %%
from PIL.ImageFont import MAX_STRING_LENGTH
from IPython import get_ipython

_ip = get_ipython()
if _ip:
    _ip.run_line_magic("load_ext", "autoreload")
    _ip.run_line_magic("autoreload", "2")


# %%
#
from transformers import GPT2LMHeadModel
import torch
import matplotlib.pyplot as plt


_ip.run_line_magic("matplotlib", "inline")

# %%

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")

sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)

# %%
sd_hf["transformer.wpe.weight"].view(-1)[:20]
# %%

plt.imshow(sd_hf["transformer.wpe.weight"])

# %%
plt.plot(sd_hf["transformer.wpe.weight"][:, 150])
plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
plt.plot(sd_hf["transformer.wpe.weight"][:, 250])

# %%
from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2")
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)


# %%
#
x = torch.zeros(500)
print(x.mean(), x.std())
for i in range(100):
    x += torch.randn(500)
print(x.mean(), x.std())
