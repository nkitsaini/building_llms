import abc
import typing as t

NodeLike: t.TypeAlias = t.Union["Node", float, int]


def wrap(x: NodeLike) -> "Node":
    if isinstance(x, Node):
        return x
    return Value(x)


class Node(abc.ABC):
    def __init__(self) -> None:
        self.grad: float = 0.0
        self.label = ""

    def backward(self):
        self.backprop(1.0)

    def children(self) -> "list[Node]": ...

    def backprop(self, factor: float):
        self.grad += factor
        self._child_backprop_(factor)

    @abc.abstractmethod
    def _child_backprop_(self, factor: float): ...

    @abc.abstractmethod
    def data(self) -> float: ...

    def __add__(self, other: "NodeLike") -> "Add":
        return Add(self, wrap(other))

    def __sub__(self, other: "NodeLike") -> "Add":
        return Add(self, -wrap(other))

    def __pow__(self, other: float) -> "Pow":
        return Pow(self, other)

    def __neg__(self) -> "ScalarMul":
        return ScalarMul(self, -1)

    def __mul__(self, other: "NodeLike") -> "Mul":
        return Mul(self, wrap(other))

    def __truediv__(self, other: "NodeLike") -> "Div":
        return Div(self, wrap(other))

    def relu(self) -> "Relu":
        return Relu(self)

    def reset_grad(self):
        self.grad = 0.0


@t.final
class Relu(Node):
    def __init__(self, a: "Node") -> None:
        self.a = a
        super().__init__()

    def data(self) -> float:
        return max(self.a.data(), 0)

    def _child_backprop_(self, factor: float):
        data = self.a.data()
        if data <= 0:
            self.a.backprop(0)
        else:
            self.a.backprop(factor)

    def reset_grad(self):
        self.a.reset_grad()


@t.final
class Div(Node):
    def __init__(self, a: "Node", b: "Node") -> None:
        self.a = a
        self.b = b
        super().__init__()

    def data(self) -> float:
        return self.a.data() / self.b.data()

    def _child_backprop_(self, factor: float):
        self.a.backprop(factor * 1 / self.b.data())
        self.b.backprop(factor * self.a.data() * -1 / (self.b.data() ** 2))

    def reset_grad(self):
        self.a.reset_grad()
        self.b.reset_grad()


@t.final
class Value(Node):
    def __init__(self, value: float) -> None:
        self.value = value
        super().__init__()

    def data(self) -> float:
        return self.value

    def _child_backprop_(self, factor: float):
        return


@t.final
class Mul(Node):
    def __init__(self, a: Node, b: Node) -> None:
        self.a = a
        self.b = b
        super().__init__()

    def data(self) -> float:
        return self.a.data() * self.b.data()

    def _child_backprop_(self, factor: float):
        self.a.backprop(self.b.data() * factor)
        self.b.backprop(self.a.data() * factor)

    def reset_grad(self):
        self.a.reset_grad()
        self.b.reset_grad()


@t.final
class ScalarMul(Node):
    def __init__(self, n: Node, scalar: float) -> None:
        self.n = n
        self.scalar = scalar
        super().__init__()

    def data(self) -> float:
        return self.n.data() * self.scalar

    def _child_backprop_(self, factor: float):
        self.n.backprop(self.scalar * factor)

    def reset_grad(self):
        self.n.reset_grad()


@t.final
class Pow(Node):
    def __init__(self, a: Node, pow: float) -> None:
        self.a = a
        self.pow = pow
        super().__init__()

    def data(self) -> float:
        return self.a.data() ** self.pow

    def _child_backprop_(self, factor: float):
        self.a.backprop(self.pow * (self.a.data() ** (self.pow - 1)) * factor)

    def reset_grad(self):
        self.a.reset_grad()


@t.final
class Add(Node):
    def __init__(self, a: Node, b: Node) -> None:
        self.a = a
        self.b = b
        super().__init__()

    def data(self) -> float:
        return self.a.data() + self.b.data()

    def _child_backprop_(self, factor: float):
        self.a.backprop(factor)
        self.b.backprop(factor)

    def reset_grad(self):
        self.a.reset_grad()
        self.b.reset_grad()
