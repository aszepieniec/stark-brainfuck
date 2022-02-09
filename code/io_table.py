from table import *


class IOTable(Table):
    # name column
    column = 0

    width = 1

    def __init__(self, field, height, generator, order):
        super(IOTable, self).__init__(field, 1, height, generator, order)

    def pad(self):
        assert(False), "You have no idea what you're doing."

    def transition_constraints(self):
        return []

    def boundary_constraints(self):
        return []
