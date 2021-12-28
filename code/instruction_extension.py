from instruction_table import *

class InstructionExtension(InstructionTable):
    def __init__( self, challenges ):
        field = challenges[0].field

        # names for challenges
        challenges = [MPolynomial.constant(c) for c in challenges]
        self.a = challenges[0]
        self.b = challenges[1]
        self.c = challenges[2]
        self.alpha = challenges[6]

        super(InstructionExtension, self).__init__(field)
        self.width = 2+2

    def transition_constraints(self):
        address, instruction, permutation, subset, \
             address_next, instruction_next, permutation_next, subset_next = MPolynomial.variables(8, self.field)
        
        polynomials = InstructionExtension.transition_constraints_afo_named_variables(address, instruction, address_next, instruction_next)

        polynomials += [permutation * \
                            ( self.alpha - self.a * address \
                                - self.b * instruction \
                                - self.c * instruction_next ) \
                             - permutation_next]

        ifnewaddress = address_next - address
        ifoldaddress = address_next - address - MPolynomial.constant(self.field.one())

        polynomials += [ifnewaddress * \
                            ( \
                                subset * \
                                ( \
                                    self.eta \
                                    - self.a * address \
                                    - self.b * instruction \
                                ) \
                                - subset_next \
                            ) \
                        + ifoldaddress * \
                            ( \
                                subset - subset_next
                            )]

        return polynomials
    
    def boundary_constraints(self):
        # format: (column, row, value)
        return [(0, 0, BaseFieldElement(0, self.field)), # count starts at zero
                (2, 0, BaseFieldElement(1, self.field))] # running product starts at one