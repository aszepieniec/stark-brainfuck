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
        for (register, cycle, value) in self.boundary_constraints():
            assert(self.table[cycle][register] == value), "boundary constraint not satisfied"

        transition_constraints = self.transition_constraints()
        for i in range(len(transition_constraints)):
            mpo = transition_constraints[i]
            for rowidx in range(self.nrows()-1):
                point = self.table[rowidx] + self.table[rowidx+1]
                assert(mpo.evaluate(point).is_zero()), f"transition constraint {i} not satisfied in row {rowidx}; point: {[p.value for p in point]}"
