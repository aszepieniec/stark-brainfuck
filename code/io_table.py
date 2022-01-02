from aet import *

class IOTable(Table):
    def __init__(self, field):
        super(IOTable, self).__init__(field, 1)

    def transition_constraints(self):
        return []
    
    def boundary_constraints(self):
        return []
        