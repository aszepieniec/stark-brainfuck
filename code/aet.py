from abc import abstractmethod
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
            row, mpo = self.boundary_constraints()[i]
            if len(self.table) != 0:
                point = self.table[row]
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

    def interpolate( self, omega, order, num_randomizers ):
        return self.interpolate_columns(omega, order, num_randomizers, columns=range(self.width))

    def interpolate_columns( self, omega, order, num_randomizers, columns ):
        num_rows = len(self.table)
        self.domain_length = 1
        self.omicron = omega
        while self.domain_length < num_rows + num_randomizers:
            self.domain_length = self.domain_length << 1
        while order > self.domain_length:
            self.omicron = self.omicron^2
            order = order / 2

        polynomials = []
        for i in columns:
            trace = [self.field.sample(os.urandom(8)) for j in range(self.domain_length)]
            for j in range(num_rows):
                trace[2*j] = self.table[i][j]
            polynomials += [intt(self.omicron, trace)]

        self.polynomials = polynomials

        return polynomials