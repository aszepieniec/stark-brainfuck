from instruction_table import *
from table_extension import TableExtension


class InstructionExtension(TableExtension):
    # name columns
    address = 0
    current_instruction = 1
    next_instruction = 2

    permutation = 3
    evaluation = 4

    width = 5

    def __init__(self, length, num_randomizers, generator, order, a, b, c, alpha, eta, permutation_terminal, evaluation_terminal):
        super(InstructionExtension, self).__init__(
            a.field, InstructionTable.width, InstructionExtension.width, length, num_randomizers, generator, order)

        # terminal values (placeholders)
        self.permutation_terminal = self.field.zero()
        self.evaluation_terminal = self.field.zero()

        # names for challenges
        self.a = MPolynomial.constant(a)
        self.b = MPolynomial.constant(b)
        self.c = MPolynomial.constant(c)
        self.alpha = MPolynomial.constant(alpha)
        self.eta = MPolynomial.constant(eta)
        self.challenges = [a, b, c, alpha, eta]

        # terminals
        self.permutation_terminal = permutation_terminal
        self.evaluation_terminal = evaluation_terminal
        self.terminals = [permutation_terminal, evaluation_terminal]

    @staticmethod
    def prepare_verify(log_num_rows, challenges, terminals):
        a, b, c, alpha, eta = challenges
        instruction_extension = InstructionExtension(
            a, b, c, alpha, eta, terminals[0], terminals[1])

        instruction_extension.log_num_rows = log_num_rows
        return instruction_extension

    @staticmethod
    def extend(instruction_table, a, b, c, alpha, eta, processor_instruction_permutation_initial):
        # algebra stuff
        field = instruction_table.field
        xfield = a.field
        one = xfield.one()
        zero = xfield.zero()

        # prepare loop
        table_extension = []
        permutation_running_product = processor_instruction_permutation_initial
        evaluation_running_sum = zero
        previous_address = -one

        # loop over all rows of table
        for i in range(len(instruction_table.matrix)):
            row = instruction_table.matrix[i]
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # permutation argument
            new_row += [permutation_running_product]
            if not new_row[InstructionExtension.current_instruction].is_zero():
                permutation_running_product *= alpha - \
                    a * new_row[InstructionExtension.address] - \
                    b * new_row[InstructionExtension.current_instruction] - \
                    c * new_row[InstructionExtension.next_instruction]
                # print("%i:" % i, permutation_running_product)

            # evaluation argument
            if new_row[InstructionExtension.address] != previous_address:
                evaluation_running_sum = eta * evaluation_running_sum + \
                    a * new_row[InstructionExtension.address] + \
                    b * new_row[InstructionExtension.current_instruction] + \
                    c * new_row[InstructionExtension.next_instruction]
            new_row += [evaluation_running_sum]

            previous_address = new_row[InstructionExtension.address]

            table_extension += [new_row]

        extended_instruction_table = InstructionExtension(
            instruction_table.length, instruction_table.num_randomizers, instruction_table.generator, instruction_table.order, a, b, c, alpha, eta, permutation_running_product, evaluation_running_sum)
        extended_instruction_table.matrix = table_extension

        extended_instruction_table.field = xfield

        extended_instruction_table.polynomials = instruction_table.polynomials
        extended_instruction_table.codewords = [
            [xfield.lift(c) for c in cdwd] for cdwd in instruction_table.codewords]

        return extended_instruction_table

    def transition_constraints_ext(self, challenges):
        a, b, c, alpha, eta = [MPolynomial.constant(c) for c in challenges]
        address, current_instruction, next_instruction, permutation, evaluation, \
            address_next, current_instruction_next, next_instruction_next, permutation_next, evaluation_next = MPolynomial.variables(
                2*self.width, self.field)

        polynomials = InstructionTable.transition_constraints_afo_named_variables(
            address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next)

        assert(len(polynomials) ==
               3), f"expected to inherit 3 polynomials from ancestor but got {len(polynomials)}"

        polynomials += [(permutation *
                         (alpha
                          - a * address
                          - b * current_instruction
                          - c * next_instruction)
                        - permutation_next) * current_instruction]

        ifnewaddress = address_next - address
        ifoldaddress = address_next - address - \
            MPolynomial.constant(self.field.one())

        polynomials += [ifnewaddress *
                        (
                            evaluation * eta
                            + a * address_next
                            + b * current_instruction_next
                            + c * next_instruction_next
                            - evaluation_next
                        )
                        + ifoldaddress *
                        (
                            evaluation - evaluation_next
                        )]

        return polynomials

    def boundary_constraints_ext(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [x[self.address] - zero,  # address starts at zero
                # x[self.permutation] - one,  # running product starts at 1
                x[self.evaluation] - self.a * x[self.address] - self.b * x[self.current_instruction] - self.c * x[self.next_instruction]]

    def terminal_constraints_ext(self, challenges, terminals):
        a, b, c, alpha, eta = challenges
        permutation_terminal, evaluation_terminal = terminals
        x = MPolynomial.variables(self.width, a.field)
        one = MPolynomial.constant(a.field.one())
        zero = MPolynomial.zero()

        constraints = []

        # polynomials += [(permutation * \
        #                     ( alpha \
        #                         - a * address  \
        #                         - b * current_instruction \
        #                         - c * next_instruction ) \
        #                 - permutation_next) * current_instruction]
        constraints += [x[InstructionExtension.current_instruction]]

        # TODO: Isn't there a constraint missing here relating to the permutation terminal?

        # polynomials += [ifnewaddress *
        #         (
        #             evaluation * eta
        #             + a * address_next
        #             + b * current_instruction_next
        #             + c * next_instruction_next
        #             - evaluation_next
        #         )
        #         + ifoldaddress *
        #         (
        #             evaluation - evaluation_next
        #         )]
        constraints += [x[InstructionExtension.evaluation] -
                        MPolynomial.constant(evaluation_terminal)]

        return constraints
