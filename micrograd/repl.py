# %%
from IPython import get_ipython

_ip = get_ipython()
if _ip:
    _ip.run_line_magic("load_ext", "autoreload")
    _ip.run_line_magic("autoreload", "2")


# %% Imports - external libs

from IPython.display import display, Image
import matplotlib.pyplot as plt
from collections.abc import Sequence

_ip.run_line_magic("matplotlib", "inline")

import numpy as np

# %% Imports - our code
from micrograd.engine import MLP, Neuron, Node, NodeLike
from micrograd.render import draw_node

# %%

a = Node(3, label="a")
b = Node(2, label="b")
c = a * b
c.label = "c"
d = c + 3 + b
d.label = "d"
e = d + c
e.label = "e"
f = d + e
f.label = "f"
g = e + f
g.label = "g"

draw_node(g, "TB")
# %%
g.reset_grad()
g.backprop()

draw_node(g)

# %% Plt
plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2)))
plt.grid()

# %% Cell 2

# inputs
x1 = Node(2.0, label="x1", calc_grad=False)
x2 = Node(0.0, label="x2", calc_grad=False)

# weights
w1 = Node(-3.0, label="w1")
w2 = Node(1.0, label="w2")

# bias
b = Node(6.8813735, label="b")

# x1w1 + x2w2 + b
x1w1 = x1 * w1
x1w1.label = "x1w1"
x2w2 = x2 * w2
x2w2.label = "x2w2"

n = x1w1 + x2w2 + b
n.label = "n"
o = n.tanh()
o.label = "o"

# %%
o.backprop()

# %%
draw_node(o)

# %%
#
n = Neuron(2)
x = [2.0, 3.0]
o = n(x)
draw_node(o)

# %%
#
x = [2.0, 3.0, -1.0]
mlp = MLP(3, [4, 4, 1])

mlp(x)
# draw_node(mlp(x)[0])

# %%


def loss(ys: Sequence[NodeLike], ypred: Sequence[Node]) -> Node:
    assert len(ys) == len(ypred)
    assert len(ys) > 0
    l = [(yp - y) ** 2 for y, yp in zip(ys, ypred)]
    result = sum(l)
    assert result != 0
    return result


def update_grad(params: list[Node], lr: float = 1e-3):
    for param in params:
        param.data -= lr * param.grad


def training_loop(
    xs: Sequence[Sequence[NodeLike]], ys: Sequence[NodeLike], mlp: MLP, lr: float = 1e-3
):
    ypred = [mlp.one(x) for x in xs]
    l = loss(ys, ypred)
    l.reset_grad()
    l.backprop()
    update_grad(mlp.parameters(), lr)


# %%
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]
ypred = [mlp.one(x) for x in xs]
l = loss(ys, ypred)
ypred, l
# %%
l.reset_grad()
l.backprop()

# %%
draw_node(mlp.layers[0].neurons[0].w[0])

# %%
#
update_grad(mlp.parameters(), 1e-2)

# %%
for _ in range(20000):
    training_loop(xs, ys, mlp, 1e-3)
