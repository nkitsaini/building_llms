from pytest import approx

from .engine import *  # noqa: F403


def test_grad():
    a = Node(1.0)
    b = Node(2.0)
    c = a + b  # 3.0

    # d = (a + b)*2
    d = c * 2  # 6.0

    # e = ((a+b)*2)**3
    # de/da = 3 ((a+b)*2)**2 * (2)
    e = d**3  # 6.0

    assert d.data == 6.0

    d.backprop()
    assert a.grad == 2

    d.reset_grad()
    e.backprop()

    assert 216 == 3 * ((1 + 2) * 2) ** 2 * 2
    assert a.grad == 216 == 3 * ((1 + 2) * 2) ** 2 * 2


def test_div():
    a = Node(4.0)
    b = Node(1.0)
    c = a / b
    assert c.data == approx(4.0)
    c.backprop()
    assert a.grad == approx(1.0)
    assert b.grad == approx(-4.0)


def test_pow():
    a = Node(4.0)
    c = a**3
    assert c.data == approx(64)
    c.backprop()
    assert a.grad == approx(3 * 4**2)


def test_sub():
    a = Node(4.0)
    b = Node(1.0)
    c = a - b
    assert c.data == approx(3)
    c.backprop()
    assert a.grad == approx(1)
    assert b.grad == approx(-1)


def test_tanh():
    a = Node(0.7)
    atan = a.tanh()
    b = Node(0.7)
    btan = b.tanh_via_exp()
    assert atan.data == approx(btan.data) == approx(0.604367777)
    atan.backprop()
    btan.backprop()
    assert a.grad == approx(b.grad)


def test_grad_micrograd_plan():
    a = Node(-4.0)
    b = Node(2.0)
    c = a + b
    d = a * b + b**3
    c += c + Node(1)
    c += Node(1) + c + (-a)
    d += d * Node(2) + (b + a).relu()
    d += Node(3) * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / Node(2.0)
    g += Node(10.0) / f

    assert g.data == approx(24.7041)
    g.backprop()
    assert b.grad == approx(645.5773)
    assert a.grad == approx(138.8338)
