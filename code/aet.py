from abc import abstractmethod
from random import random
from multivariate import *
from ntt import *
import os


class Table:
    def __init__(self, field, width):
        self.field = field
        self.width = width
        self.table = []

    def nrows(self):
        return len(self.table)

    def ncols(self):
        return self.width

    # @abstractmethod
    # @staticmethod
    # def transition_constraints():
    #     pass

    # @abstractmethod
    # @staticmethod
    # def boundary_constraints():
    #     pass

    def test(self):
        for i in range(len(self.boundary_constraints())):
            mpo = self.boundary_constraints()[i]
            if len(self.table) != 0:
                point = self.table[0]
                assert(mpo.evaluate(point).is_zero(
                )), f"BOUNDARY constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"

        transition_constraints = self.transition_constraints()
        for i in range(len(transition_constraints)):
            mpo = transition_constraints[i]
            for rowidx in range(self.nrows()-1):
                assert(len(self.table[rowidx]) == len(
                    self.table[rowidx+1])), "table has consecutive rows of different length"
                point = self.table[rowidx] + self.table[rowidx+1]
                assert(len(point) == len(list(mpo.dictionary.keys())[
                       0])), f"point has {len(point)} elements but mpo has {len(list(mpo.dictionary.keys())[0])} variables .."
                assert(mpo.evaluate(point).is_zero(
                )), f"TRNASITION constraint {i} not satisfied in row {rowidx}; point: {[str(p) for p in point]}; polynomial {str(mpo.partial_evaluate({1: point[1]}))} evaluates to {str(mpo.evaluate(point))}"

    def interpolate(self, omega, order, num_randomizers):
        return self.interpolate_columns(omega, order, num_randomizers, column_indices=range(self.width))

    def interpolate_columns(self, omega, order, num_randomizers, column_indices):
        num_rows = len(self.table)
        if num_rows == 0:
            return [Polynomial([self.field.zero()])] * self.width

        assert(num_rows != 0), "number of rows cannot be zero"

        assert(num_rows & (num_rows - 1) ==
               0), f"num_rows has value {num_rows} but must be a power of two"

        assert(omega ^ order == omega.field.one()
               ), "order must match with omega"
        assert(omega ^ (order//2) != omega.field.one()
               ), "order must be primitive"

        self.domain_length = 1
        while self.domain_length < num_rows + num_randomizers:
            self.domain_length = self.domain_length << 1

        randomness_expansion_factor = int(self.domain_length / num_rows)

        self.omicron = omega
        while order > self.domain_length:
            self.omicron = self.omicron ^ 2
            order = order // 2

        assert(self.omicron ^ order == self.omicron.field.one()
               ), "order must now be order of omicron"
        assert(self.omicron ^ (order//2) !=
               self.omicron.field.one()), "order not primitive"
        assert(self.domain_length == order)

        polynomials = []
        for i in column_indices:
            trace = [self.field.sample(os.urandom(3*8))
                     for j in range(order)]
            for j in range(num_rows):
                trace[randomness_expansion_factor*j] = self.table[j][i]
            polynomials += [Polynomial(intt(self.field.lift(self.omicron), trace))]

        return polynomials
