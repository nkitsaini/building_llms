import abc
import enum
import math
import random
import typing as t
from collections.abc import Sequence
from dataclasses import dataclass

NodeLike: t.TypeAlias = t.Union["Node", float, int]


def wrap(x: NodeLike) -> "Node":
    if isinstance(x, Node):
        return x
    return Node(x, label="-", calc_grad=False)


class Op(enum.Enum):
    # empty
    Noop = enum.auto()
    # a + b
    Add = enum.auto()
    # -a
    Neg = enum.auto()
    # a**b
    Pow = enum.auto()
    # a*b
    Mul = enum.auto()
    # a/b
    Div = enum.auto()
    # max(0,a)
    Relu = enum.auto()
    # a
    Tanh = enum.auto()
    # e^a
    Exp = enum.auto()

    def validate(self, children: "tuple[Node, ...]" = ()):
        assert self.length() == len(children), (
            f"Children count ({len(children)}) does not match what the operation ({self}) expects"
        )

    def length(self):
        match self:
            case self.Noop:
                return 0
            case self.Add:
                return 2
            case self.Neg:
                return 1
            case self.Pow:
                return 2
            case self.Mul:
                return 2
            case self.Div:
                return 2
            case self.Relu:
                return 1
            case self.Tanh:
                return 1
            case self.Exp:
                return 1

    def is_noop(self) -> bool:
        return self == Op.Noop

    def render(self) -> str:
        match self:
            case self.Noop:
                return "NO_OP"
            case self.Add:
                return "+"
            case self.Neg:
                return "-"
            case self.Pow:
                return "**"
            case self.Mul:
                return "*"
            case self.Div:
                return "/"
            case self.Relu:
                return "ReLU"
            case self.Tanh:
                return "tanh"
            case self.Exp:
                return "exp"


@t.final
class Node:
    def __init__(
        self,
        data: float,
        children: "tuple[Node, ...]" = (),
        op: "Op" = Op.Noop,
        label: str = "",
        # If false, this node will not be treated as a variable when backpropogating
        calc_grad: bool = True,
    ) -> None:
        op.validate(children)
        self.data = data
        self.children = children
        self.op = op
        self.grad: float = 0.0
        self.label = label
        self.calc_grad = calc_grad

    def backprop(self, factor: float = 1.0):
        if not self.calc_grad:
            return

        self.grad += factor
        match self.op:
            case Op.Add:
                self.children[0].backprop(factor)
                self.children[1].backprop(factor)
            case Op.Neg:
                self.children[0].backprop(-factor)
            case Op.Pow:
                a, x = self.children
                self.children[0].backprop(
                    factor * x.data * math.pow(a.data, x.data - 1)
                )

                # safity net, otherwise `math.log()` will fail
                if self.children[1].calc_grad:
                    self.children[1].backprop(
                        factor * math.pow(a.data, x.data) * math.log(a.data)
                    )
            case Op.Mul:
                a, b = self.children
                a.backprop(factor * b.data)
                b.backprop(factor * a.data)
            case Op.Div:
                a, b = self.children
                a.backprop(factor * 1 / b.data)
                b.backprop(factor * a.data * -1 / (b.data**2))
            case Op.Relu:
                a = self.children[0]
                if a.data > 0:
                    a.backprop(factor)
            case Op.Exp:
                a = self.children[0]
                a.backprop(factor * (self.data))
            case Op.Tanh:
                a = self.children[0]
                a.backprop(factor * (1 - self.data**2))
            case Op.Noop:
                pass

    def reset_grad(self):
        self.grad = 0.0
        for child in self.children:
            child.reset_grad()

    def __add__(self, other: "NodeLike") -> "Node":
        other = wrap(other)
        return Node(self.data + other.data, children=(self, other), op=Op.Add)

    def __radd__(self, other: "NodeLike") -> "Node":
        other = wrap(other)
        return Node(self.data + other.data, children=(other, self), op=Op.Add)

    def __neg__(self) -> "Node":
        return Node(-self.data, children=(self,), op=Op.Neg)

    def __sub__(self, other: "NodeLike") -> "Node":
        return self + (-wrap(other))

    def __pow__(self, other: "Node | float") -> "Node":
        other = wrap(other)

        # variable_pow for gradable powers
        assert not other.calc_grad
        return Node(math.pow(self.data, other.data), children=(self, other), op=Op.Pow)

    def variable_pow(self, other: "Node | float") -> "Node":
        assert self.data >= 0, "Cannot grad a^x when a < 0"
        other = wrap(other)
        return Node(math.pow(self.data, other.data), children=(self, other), op=Op.Pow)

    def __mul__(self, other: "NodeLike") -> "Node":
        other = wrap(other)
        return Node(self.data * other.data, children=(self, other), op=Op.Mul)

    def __truediv__(self, other: "NodeLike") -> "Node":
        other = wrap(other)
        return Node(self.data / other.data, children=(self, other), op=Op.Div)

    def relu(self) -> "Node":
        return Node(max(0, self.data), children=(self,), op=Op.Relu)

    def exp(self) -> "Node":
        return Node(math.exp(self.data), children=(self,), op=Op.Exp)

    def tanh_via_exp(self) -> "Node":
        ex = self.exp()
        ex_neg = (-self).exp()
        return (ex - ex_neg) / (ex + ex_neg)

    def tanh(self) -> "Node":
        ex = math.exp(self.data)
        enegx = math.exp(-self.data)
        return Node((ex - enegx) / (ex + enegx), children=(self,), op=Op.Tanh)

    @t.override
    def __repr__(self) -> str:
        return (
            f"Node(data={self.data:.4f}, grad={self.grad:.4f}, op={self.op.render()})"
        )


@t.final
class Neuron:
    def __init__(self, input_count: int) -> None:
        self.w = [Node(random.uniform(-1, 1)) for _ in range(input_count)]
        self.b = Node(random.uniform(-1, 1))

    def __call__(self, inputs: "Sequence[NodeLike]") -> "Node":
        assert len(inputs) == len(self.w), f"{len(inputs)=} {len(self.w)=}"
        result = self.b
        for w, x in zip(self.w, inputs):
            result += w * x
        return result.tanh()

    def parameters(self) -> "list[Node]":
        return self.w + [self.b]


@t.final
class Layer:
    def __init__(self, input_count: int, output_count: int) -> None:
        self.neurons = [Neuron(input_count) for _ in range(output_count)]

    def __call__(self, inputs: "Sequence[NodeLike]") -> "list[Node]":
        result = [n(inputs) for n in self.neurons]
        return result

    def parameters(self) -> "list[Node]":
        params: list[Node] = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


@t.final
class MLP:
    def __init__(self, input_count: int, output_counts: list[int]) -> None:
        assert len(output_counts) > 0
        self.layers = [
            Layer(i, o) for i, o in zip([input_count] + output_counts, output_counts)
        ]

    def __call__(self, inputs: "Sequence[NodeLike]") -> "list[Node]":
        output = self.layers[0](inputs)  # fixes type
        for layer in self.layers[1:]:
            output = layer(output)
        return output

    def one(self, inputs: "Sequence[NodeLike]") -> "Node":
        output = self(inputs)
        assert len(output) == 1
        return output[0]

    def parameters(self) -> "list[Node]":
        params: list[Node] = []
        for n in self.layers:
            params.extend(n.parameters())
        return params
