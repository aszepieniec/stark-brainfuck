from aet import *

class InstructionTable(Table):
    def __init__(self, field):
        super(InstructionTable, self).__init__(field, 2)
    
    @staticmethod
    def transition_constraints_afo_named_variables( address, instruction, address_next, instruction_next ):
        field = address.coefficients.values()[0].field
        one = MPolynomial.constant(field.one())

        polynomials = []
        polynomials += [(address_next - address - one) * (address_next - address)] # instruction pointer increases by 0 or 1
        polynomials += [(address_next - address - one) * (instruction_next - instruction)] # if address is the same, then instruction is also
        
        return polynomials

    def transition_constraints( self ):
        address, instruction, address_next, instruction_next = MPolynomial.variables(4, self.field)
        return InstructionTable.transition_constraints_afo_named_variables(address, instruction, address_next, instruction_next)

    def boundary_constraints( self ):
        # format: (column, row, value)
        return [(0, 0, BaseFieldElement(0, self.field))] # count starts at zero
