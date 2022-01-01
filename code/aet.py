from abc import abstractmethod
from multivariate import *


class Table:
    def __init__(self, field, width):
        self.field = field
        self.width = width
        self.table = []

    def nrows(self):
        return len(self.table)

    def ncols(self):
        return self.width

    @abstractmethod
    def transition_constraints(self):
        pass

    @abstractmethod
    def boundary_constraints(self):
        pass

    def test(self):
        for i in range(len(self.boundary_constraints())):
            cycle, mpo = self.boundary_constraints()[i]
            if len(self.table) != 0:
                point = self.table[cycle]
                assert(mpo.evaluate(point).is_zero()), f"BOUNDARY constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"

        transition_constraints = self.transition_constraints()
        for i in range(len(transition_constraints)):
            mpo = transition_constraints[i]
            for rowidx in range(self.nrows()-1):
                assert(len(self.table[rowidx]) == len(self.table[rowidx+1])), "table has consecutive rows of different length"
                point = self.table[rowidx] + self.table[rowidx+1]
                assert(len(point) == len(list(mpo.dictionary.keys())[0])), f"point has {len(point)} elements but mpo has {len(list(mpo.dictionary.keys())[0])} variables .."
                assert(mpo.evaluate(point).is_zero()), f"TRNASITION constraint {i} not satisfied in row {rowidx}; point: {[str(p) for p in point]}; polynomial {str(mpo.partial_evaluate({2: point[2]}))} evaluates to {str(mpo.evaluate(point))}"
