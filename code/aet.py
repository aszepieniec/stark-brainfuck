from abc import abstractmethod
from multivariate import *


class Table:
    def __init__(self, field, base_width, extension_width):
        self.field = field
        self.base_width = base_width
        self.extension_width = extension_width
        self.table = []

    def nrows(self):
        return len(self.table)

    def ncols_base(self):
        return self.base_width

    def ncols_extension(self):
        return self.extension_width

    @abstractmethod
    def transition_constraints(self):
        pass

    @abstractmethod
    def boundary_constraints(self):
        pass

    @abstractmethod
    def extended_transition_constraints(self):
        pass

    @abstractmethod
    def extended_boundary_constraints(self):
        pass

    @abstractmethod
    def test_base(self):
        pass

    @abstractmethod
    def test_extension(self):
        pass
