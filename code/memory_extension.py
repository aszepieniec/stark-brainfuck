from memory_table import *

class MemoryExtension(MemoryTable):
    def __init__(self, challenges ):
        field = challenges[0].field

        # names for challenges
        challenges = [MPolynomial.constant(c) for c in challenges]

        self.d = challenges[3]
        self.e = challenges[4]
        self.f = challenges[5]
        self.beta = challenges[7]

        super(MemoryTable,self).__init__(field,3+1)
    
    def transition_constraints(self):
        cycle, address, value, permutation, \
            cycle_next, address_next, value_next, permutation_next = MPolynomial.variables(8, self.field)
            
        polynomials = MemoryTable.transition_constraints_afo_named_variables(cycle, address, value, cycle_next, address_next, value_next)

        polynomials += [permutation * \
                            ( self.beta - self.d * cycle \
                                - self.e * address \
                                 - self.f * value ) \
                         - permutation_next]

        return polynomials
    
    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[0] - zero),  # cycle
                (0, x[1] - zero),  # memory pointer
                (0, x[2] - zero),  # memory value
                (0, x[3] - one),   # permutation
                ]