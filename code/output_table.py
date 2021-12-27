from aet import *

class OutputTable(Table):
    def __init__(self, field):
        super(OutputTable, self).__init__(field, 1)

    def transition_constraints(self):
        return []
    
    def boundary_constraints(self):
        return []
        