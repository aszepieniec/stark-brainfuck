from output_table import *

class OutputExtension(OutputTable):
    def __init__(self, challenges):
        field = challenges[0].field

        # names for challenges
        challenges = [MPolynomial.constant(c) for c in challenges]
        self.delta = challenges[9]

        super(OutputExtension, self).__init__(field)
        self.width = 1+2

    def transition_constraints(self):
        output, indeterminate, evaluation, \
            output_next, indeterminate_next, evaluation_next = MPolynomial.variables(6, self.field)
        
        polynomials = []

        polynomials += [indeterminate_next - indeterminate * self.delta]
        polynomials += [evaluation + indeterminate * output - evaluation_next]
        
        return polynomials
    
    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[1] - one), # indeterminate
                (0, x[2] - zero)] # evaluation