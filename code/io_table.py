from aet import *


class IOTable(Table):
    # name column
    column = 0

    width = 1

    def __init__(self, field):
        super(IOTable, self).__init__(field, 1)

    def pad(self):
        assert(False), "You have no idea what you're doing."

    def transition_constraints(self):
        return []

    def boundary_constraints(self):
        return []
