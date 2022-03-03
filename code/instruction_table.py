from processor_table import ProcessorTable
from table import *


class InstructionTable(Table):
    # named indices for base columns
    address = 0
    current_instruction = 1
    next_instruction = 2

    # named indices for extension columns
    permutation = 3
    evaluation = 4

    def __init__(self, field, length, num_randomizers, generator, order):
        super(InstructionTable, self).__init__(
            field, 3, 5, length, num_randomizers, generator, order)

    def pad(self):
        while len(self.matrix) & (len(self.matrix)-1):
            new_row = [self.field.zero()] * self.base_width
            new_row[InstructionTable.address] = self.matrix[-1][InstructionTable.address]
            new_row[InstructionTable.current_instruction] = self.field.zero()
            new_row[InstructionTable.next_instruction] = self.field.zero()
            self.matrix += [new_row]

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

    def base_transition_constraints(self):
        address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next = MPolynomial.variables(
            6, self.field)
        return InstructionTable.transition_constraints_afo_named_variables(address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next)

    def base_boundary_constraints(self):
        # format: mpolynomial
        x = MPolynomial.variables(self.width, self.field)
        zero = MPolynomial.zero()
        return [x[InstructionTable.address]-zero]

      #
    # # #
      #

    def transition_constraints_ext(self, challenges):
        field = challenges[0].field
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [MPolynomial.constant(ch) for ch in challenges]
        address, current_instruction, next_instruction, permutation, evaluation, \
            address_next, current_instruction_next, next_instruction_next, permutation_next, evaluation_next = MPolynomial.variables(
                2*self.full_width, field)

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
            MPolynomial.constant(field.one())

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

    def boundary_constraints_ext(self, challenges):
        field = challenges[0].field
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [MPolynomial.constant(ch) for ch in challenges]
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.full_width, field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [x[InstructionTable.address] - zero,  # address starts at zero
                # x[self.permutation] - one,  # running product starts at 1
                x[InstructionTable.evaluation] - 
                a * x[InstructionTable.address] - 
                b * x[InstructionTable.current_instruction] - 
                c * x[InstructionTable.next_instruction]]

    def terminal_constraints_ext(self, challenges, terminals):
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [MPolynomial(ch) for ch in challenges]
        processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal, instruction_evaluation_terminal = terminals
        field = challenges[0].field
        x = MPolynomial.variables(self.full_width, field)

        constraints = []

        # polynomials += [(permutation * \
        #                     ( alpha \
        #                         - a * address  \
        #                         - b * current_instruction \
        #                         - c * next_instruction ) \
        #                 - permutation_next) * current_instruction]
        constraints += [x[InstructionTable.current_instruction]]

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
        constraints += [x[InstructionTable.evaluation] -
                        MPolynomial.constant(instruction_evaluation_terminal)]

        return constraints

    def extend(self, all_challenges, all_initials):
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = all_challenges
        processor_instruction_permutation_initial, processor_memory_permutation_initial = all_initials
        # algebra stuff
        field = self.field
        xfield = a.field
        one = xfield.one()
        zero = xfield.zero()

        # prepare loop
        extended_matrix = []
        permutation_running_product = processor_instruction_permutation_initial
        evaluation_running_sum = zero
        previous_address = -one

        # loop over all rows of table
        for i in range(len(self.matrix)):
            row = self.matrix[i]
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # permutation argument
            new_row += [permutation_running_product]
            if not new_row[InstructionTable.current_instruction].is_zero():
                permutation_running_product *= alpha - \
                    a * new_row[InstructionTable.address] - \
                    b * new_row[InstructionTable.current_instruction] - \
                    c * new_row[InstructionTable.next_instruction]
                # print("%i:" % i, permutation_running_product)

            # evaluation argument
            if new_row[InstructionTable.address] != previous_address:
                evaluation_running_sum = eta * evaluation_running_sum + \
                    a * new_row[InstructionTable.address] + \
                    b * new_row[InstructionTable.current_instruction] + \
                    c * new_row[InstructionTable.next_instruction]
            new_row += [evaluation_running_sum]

            previous_address = new_row[InstructionTable.address]

            extended_matrix += [new_row]

        self.field = xfield
        self.matrix = extended_matrix
        self.codewords = [[xfield.lift(c) for c in cdwd] for cdwd in self.codewords]

        self.permutation_terminal = permutation_running_product
        self.evaluation_terminal = evaluation_running_sum
