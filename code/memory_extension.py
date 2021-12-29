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
                            ( self.beta - self.d * address_next \
                                 - self.e * value_next ) \
                         - permutation_next]

        return polynomials
    
    def boundary_constraints(self):
        return [(0, 0, self.field.zero()),  # cycle
                (1, 0, self.field.zero()),  # memory pointer
                (2, 0, self.field.zero()),  # memory value
                (3, 0, self.field.one()),   # permutation
                ]