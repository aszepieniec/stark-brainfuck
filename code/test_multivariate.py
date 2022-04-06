from algebra import *
from multivariate import MPolynomial
from extension_field import *
from univariate import *
from ntt import *
import os


def test_symbolic_bounds_zero_coefficient_zero_input():
    field = ExtensionField.main()
    linear = MPolynomial({(0, 1): field.one()})
    sym_eval = linear.symbolic_degree_bound([-1, -1])
    assert(sym_eval == -1)

    # Add a zero-coefficient and verify that the result is unchanged
    linear_alt = MPolynomial({(0, 1): field.one(), (0, 0): field.zero()})
    sym_eval_alt = linear_alt.symbolic_degree_bound([-1, -1])
    assert(sym_eval_alt == -1)
    linear_alt_alt = MPolynomial(
        {(0, 1): field.one(), (100, 42): field.zero()})
    sym_eval_alt_alt = linear_alt_alt.symbolic_degree_bound([-1, -1])
    assert(sym_eval_alt_alt == -1)
    print("Test succeeded \\0/")


def test_symbolic_bounds_zero_coefficient_non_zero_input():
    field = ExtensionField.main()
    max_degrees = [3, 3, 3]
    non_linear = MPolynomial({(0, 2, 1): field.one(), (0, 0, 1): field.one()})
    sym_eval = non_linear.symbolic_degree_bound(max_degrees)
    assert(sym_eval == 9)

    # Add a zero-coefficient and verify that the result is unchanged
    non_linear_alt = MPolynomial(
        {(0, 2, 1): field.one(), (0, 0, 1): field.one(), (3, 3, 4): field.zero()})
    sym_eval_alt = non_linear_alt.symbolic_degree_bound(max_degrees)
    assert(sym_eval_alt == 9)

    # Change to a non-zero coefficient and verify that the result has changed
    new_mpolynomial = MPolynomial(
        {(0, 2, 1): field.one(), (0, 0, 1): field.one(), (3, 3, 4): field.one()})
    sym_eval_new = new_mpolynomial.symbolic_degree_bound(max_degrees)
    assert(sym_eval_new != 9)
    print("Test succeeded \\0/")
