# %%
%load_ext autoreload
%autoreload 2

# %% Cell 1
from IPython.display import display, Image

from micrograd.engine import Node, Value
from micrograd.render import draw_node

draw_node(Value(3))


# %% Cell 2
print(4)
print(5)
