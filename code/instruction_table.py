from aet import *


class InstructionTable(Table):
    # name columns
    address = 0
    current_instruction = 1
    next_instruction = 2

    width = 3

    def __init__(self, field):
        super(InstructionTable, self).__init__(field, 3)

    def pad(self):
        while len(self.table) & (len(self.table)-1):
            new_row = [self.field.zero()] * self.width
            new_row[InstructionTable.address] = self.table[-1][InstructionTable.address]
            new_row[InstructionTable.current_instruction] = self.field.zero()
            new_row[InstructionTable.next_instruction] = self.field.zero()
            self.table += [new_row]

    @staticmethod
    def transition_constraints_afo_named_variables(address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next):
        field = list(address.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())

        polynomials = []
        # instruction pointer increases by 0 or 1
        polynomials += [(address_next - address - one)
                        * (address_next - address)]
        # if address is the same, then current instruction is also
        polynomials += [(address_next - address - one) *
                        (current_instruction_next - current_instruction)]
        # if address is the same, then next instruction is also
        polynomials += [(address_next - address - one) *
                        (next_instruction_next - next_instruction)]

        return polynomials

    def transition_constraints(self):
        address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next = MPolynomial.variables(
            6, self.field)
        return InstructionTable.transition_constraints_afo_named_variables(address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next)

    def boundary_constraints(self):
        # format: (row, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        zero = MPolynomial.zero()
        return [(0, x[InstructionTable.address]-zero)]
