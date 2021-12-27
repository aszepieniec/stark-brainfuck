from aet import *

class InputTable(Table):
    def __init__(self, field):
        super(InputTable, self).__init__(field, 1)

    def transition_constraints(self):
        return []
    
    def boundary_constraints(self):
        return []
        