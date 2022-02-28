from table import *


class IOTable(Table):
    # name column
    column = 0

    width = 1

    def __init__(self, field, length, generator, order):
        num_randomizers = 0
        super(IOTable, self).__init__(
            field, IOTable.width, length, num_randomizers, generator, order)

    def pad(self):
        self.length = len(self.matrix)
        while len(self.matrix) & (len(self.matrix) - 1) != 0:
            self.matrix += [[self.field.zero()]]
        self.height = len(self.matrix)

    def transition_constraints(self):
        return []

    def boundary_constraints(self):
        return []
