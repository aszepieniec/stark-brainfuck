from instruction_table import *


class InstructionExtension(InstructionTable):
    # name columns
    address = 0
    current_instruction = 1
    next_instruction = 2

    permutation = 3
    evaluation = 4

    def __init__(self, a, b, c, alpha, eta):
        field = a.field

        # terminal values (placeholders)
        self.permutation_terminal = field.zero()
        self.evaluation_terminal = field.zero()

        # names for challenges
        self.a = MPolynomial.constant(a)
        self.b = MPolynomial.constant(b)
        self.c = MPolynomial.constant(c)
        self.alpha = MPolynomial.constant(alpha)
        self.eta = MPolynomial.constant(eta)

        super(InstructionExtension, self).__init__(field)
        self.width = 3+1+1

    @staticmethod
    def extend(instruction_table, program, a, b, c, alpha, eta):

        # algebra stuff
        field = instruction_table.field
        xfield = a.field
        one = xfield.one()
        zero = xfield.zero()

        # prepare loop
        table_extension = []
        permutation_running_product = one
        evaluation_running_sum = zero
        previous_address = -one

        # loop over all rows of table
        for i in range(len(instruction_table.table)):
            row = instruction_table.table[i]
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # permutation argument
            new_row += [permutation_running_product]
            permutation_running_product *= alpha - \
                a * new_row[InstructionExtension.address] - \
                b * new_row[InstructionExtension.current_instruction] - \
                c * new_row[InstructionExtension.next_instruction]

            # evaluation argument
            if new_row[InstructionExtension.address] != previous_address:
                evaluation_running_sum = eta * evaluation_running_sum + \
                    a * new_row[InstructionExtension.address] + \
                    b * new_row[InstructionExtension.current_instruction] + \
                    c * new_row[InstructionExtension.next_instruction]
            new_row += [evaluation_running_sum]
            previous_address = new_row[InstructionExtension.address]

            table_extension += [new_row]

        extended_instruction_table = InstructionExtension(a, b, c, alpha, eta)
        extended_instruction_table.table = table_extension

        extended_instruction_table.permutation_terminal = permutation_running_product
        extended_instruction_table.evaluation_terminal = evaluation_running_sum

        return extended_instruction_table

    def transition_constraints(self):
        address, current_instruction, next_instruction, permutation, evaluation, \
            address_next, current_instruction_next, next_instruction_next, permutation_next, evaluation_next = MPolynomial.variables(
                2*self.width, self.field)

        polynomials = InstructionExtension.transition_constraints_afo_named_variables(
            address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next)

        polynomials += [permutation *
                        (self.alpha - self.a * address
                         - self.b * current_instruction
                                - self.c * next_instruction)
                        - permutation_next]

        ifnewaddress = address_next - address
        ifoldaddress = address_next - address - \
            MPolynomial.constant(self.field.one())

        polynomials += [ifnewaddress *
                        (
                            evaluation * self.eta
                            + self.a * address_next
                            + self.b * current_instruction_next
                            + self.c * next_instruction_next
                            - evaluation_next
                        )
                        + ifoldaddress *
                        (
                            evaluation - evaluation_next
                        )]

        return polynomials

    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[self.address] - zero),  # address starts at zero
                (0, x[self.permutation] - one),  # running product starts at 1
                (0, x[self.evaluation] - self.a * x[self.address] - self.b * x[self.current_instruction] - self.c * x[self.next_instruction])]
