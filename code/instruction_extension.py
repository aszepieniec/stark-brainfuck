from instruction_table import *


class InstructionExtension(InstructionTable):
    # names for columns
    address = 0
    instruction = 1

    permutation = 2
    evaluation = 3

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
        self.width = 2+1+1

    @staticmethod
    def extend(instruction_table, program, a, b, c, alpha, eta):

        # algebra stuff
        field = instruction_table.field
        xfield = a.field
        one = xfield.one()

        # prepare loop
        table_extension = []
        instruction_permutation_running_product = one
        subset_running_product = one

        # loop over all rows of table
        for i in range(len(instruction_table.table)):
            row = instruction_table.table[i]
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # match with this:
            # 1. running product for instruction permutation
            #instruction_permutation_running_product *= alpha - a * row[instruction_pointer] - b * row[current_instruction] - c * row[next_instruction]
            #new_row += [[instruction_permutation_running_product]]

            new_row += [instruction_permutation_running_product]
            current_instruction = new_row[InstructionExtension.instruction]
            index = row[InstructionExtension.address].value+1

            if index < len(program):
                next_instruction = xfield.lift(program[index])
            else:
                next_instruction = xfield.zero()
            instruction_permutation_running_product *= alpha - \
                a * new_row[InstructionExtension.address] - \
                b * current_instruction - \
                c * next_instruction

            # match with this

            # ifnewaddress = address_next - address
            # ifoldaddress = address_next - address - MPolynomial.constant(self.field.one())

            # polynomials += [ifnewaddress *  ( subset * ( self.eta - self.a * address - self.b * instruction ) - subset_next ) \
            #                 + ifoldaddress * ( subset - subset_next ) ]
            new_row += [subset_running_product]
            if i < len(instruction_table.table) - 1 and instruction_table.table[i+1][0] != instruction_table.table[i][0]:
                subset_running_product *= eta - a * \
                    xfield.lift(
                        instruction_table.table[i][0]) - b * xfield.lift(instruction_table.table[i][1])

            table_extension += [new_row]

        extended_instruction_table = InstructionExtension(a, b, c, alpha, eta)
        extended_instruction_table.table = table_extension

        extended_instruction_table.permutation_terminal = instruction_permutation_running_product
        extended_instruction_table.evaluation_terminal = subset_running_product

        return extended_instruction_table

    def transition_constraints(self):
        address, instruction, permutation, subset, \
            address_next, instruction_next, permutation_next, subset_next = MPolynomial.variables(
                8, self.field)

        polynomials = InstructionExtension.transition_constraints_afo_named_variables(
            address, instruction, address_next, instruction_next)

        polynomials += [permutation *
                        (self.alpha - self.a * address
                         - self.b * instruction
                                - self.c * instruction_next)
                        - permutation_next]

        ifnewaddress = address_next - address
        ifoldaddress = address_next - address - \
            MPolynomial.constant(self.field.one())

        polynomials += [ifnewaddress *
                        (
                            subset *
                            (
                                self.eta
                                - self.a * address
                                - self.b * instruction
                            )
                            - subset_next
                        )
                        + ifoldaddress *
                        (
                            subset - subset_next
                        )]

        return polynomials

    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[self.address] - zero),  # address starts at zero
                (0, x[self.permutation] - one)]  # running product starts at alpha - a * addr - b * instr - c * instr_next
