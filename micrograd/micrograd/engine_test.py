from pytest import approx

from micrograd.engine import Value

from .engine import *  # noqa: F403


def test_grad():

    a = Value(1.0)
    b = Value(2.0)
    c = a + b  # 3.0

    # d = (a + b)*2
    d = c * 2  # 6.0

    # e = ((a+b)*2)**3
    # de/da = 3 ((a+b)*2)**2 * (2)
    e = d**3  # 6.0

    assert d.data() == 6.0

    d.backward()
    assert a.grad == 2

    d.reset_grad()
    e.backward()

    assert 216 == 3 * ((1 + 2) * 2) ** 2 * 2
    assert a.grad == 216 == 3 * ((1 + 2) * 2) ** 2 * 2


def test_div():
    a = Value(4.0)
    b = Value(1.0)
    c = a / b
    assert c.data() == approx(4.0)
    c.backward()
    assert a.grad == approx(1.0)
    assert b.grad == approx(-4.0)


def test_pow():
    a = Value(4.0)
    c = a**3
    assert c.data() == approx(64)
    c.backward()
    assert a.grad == approx(3 * 4**2)


def test_sub():
    a = Value(4.0)
    b = Value(1.0)
    c = a - b
    assert c.data() == approx(3)
    c.backward()
    assert a.grad == approx(1)
    assert b.grad == approx(-1)


def test_grad_micrograd_plan():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + Value(1)
    c += Value(1) + c + (-a)
    d += d * Value(2) + (b + a).relu()
    d += Value(3) * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / Value(2.0)
    g += Value(10.0) / f

    assert g.data() == approx(24.7041)
    g.backward()
    assert b.grad == approx(645.5773)
    assert a.grad == approx(138.8338)
