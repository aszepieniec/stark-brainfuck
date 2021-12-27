from input_table import *

class InputExtension(InputTable):
    def __init__(self, challenges):
        field = challenges[0].field

        # names for challenges
        challenges = [MPolynomial.constant(c) for c in challenges]
        self.gamma = challenges[8]

        super(InputExtension, self).__init__(field)
        self.width = 1+2

    def transition_constraints(self):
        input_, indeterminate, evaluation, \
            input_next, indeterminate_next, evaluation_next = MPolynomial.variables(6, self.field)
        
        polynomials = []

        polynomials += [indeterminate_next - indeterminate * self.gamma]
        polynomials += [evaluation + indeterminate * input_ - evaluation_next]

        return polynomials
    
    def boundary_constraints(self):
        # format: (column, row, value)
        return [(1, 0, BaseFieldElement(1, self.field)), # indeterminate
                (2, 0, BaseFieldElement(0, self.field))] # evaluation