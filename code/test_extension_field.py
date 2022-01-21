import os
from extension_field import *


def test_inverse():
    field = ExtensionField.main()
    a = field.sample(os.urandom(8*3))
    assert((a * a.inverse()) == field.one()), "extension field inverse fail"


def test_xgcd():
    field = BaseField.main()
    x = Polynomial([field.sample(os.urandom(8)) for i in range(10)])
    y = Polynomial([field.sample(os.urandom(8)) for i in range(13)])

    a, b, g = Polynomial.xgcd(x, y)

    assert(a*x + y*b == g), "bezout relation fail"

    assert((a*x) % y == Polynomial([field.one()])
           ), f"inverse fail: a = {a} and x = {x} but a * x mod y = {a*x % y} =/= 1"
