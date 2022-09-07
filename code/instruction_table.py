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
        # if address changes, then current row's next instruction is the next row's current instruction
        polynomials += [(address_next - address) *
                        (next_instruction - current_instruction_next)]
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
        x = MPolynomial.variables(self.base_width, self.field)
        zero = MPolynomial.zero()
        return [x[InstructionTable.address]-zero]

      #
    # # #
      #

    @staticmethod
    def instruction_zerofier(current_instruction):
        field = list(current_instruction.dictionary.values())[0].field
        acc = MPolynomial.constant(field.one())
        for ch in ['[', ']', '<', '>', '+', '-', ',', '.']:
            ch_ = MPolynomial.constant(field(ord(ch)))
            acc *= current_instruction - ch_
        return acc

    def transition_constraints_ext(self, challenges):
        field = challenges[0].field
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(ch) for ch in challenges]
        address, current_instruction, next_instruction, permutation, evaluation, \
            address_next, current_instruction_next, next_instruction_next, permutation_next, evaluation_next = MPolynomial.variables(
                2*self.full_width, field)
        one = MPolynomial.constant(field.one())

        polynomials = InstructionTable.transition_constraints_afo_named_variables(
            address, current_instruction, next_instruction, address_next, current_instruction_next, next_instruction_next)

        polynomials += [(permutation *
                         (alpha
                          - a * address_next
                          - b * current_instruction_next
                          - c * next_instruction_next)
                         - permutation_next) *
                        current_instruction * (address + one - address_next)
                        + InstructionTable.instruction_zerofier(current_instruction) * (permutation - permutation_next) # degree 9
                        + (address - address_next) * (permutation - permutation_next)]

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
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(ch) for ch in challenges]
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
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(ch) for ch in challenges]
        processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal, instruction_evaluation_terminal = [
            MPolynomial.constant(t) for t in terminals]
        field = challenges[0].field
        x = MPolynomial.variables(self.full_width, field)
        zero = MPolynomial.zero()

        constraints = []

        # ( permutation * ( alpha \
        #                    - a * address  \
        #                    - b * current_instruction \
        #                    - c * next_instruction ) \
        #               - permutation_next) * current_instruction
        # constraints += [(x[InstructionTable.permutation] *
        #                  (alpha -
        #                   a * x[InstructionTable.address] -
        #                   b * x[InstructionTable.current_instruction] -
        #                   c * x[InstructionTable.next_instruction]) - processor_instruction_permutation_terminal) *
        #                 x[InstructionTable.current_instruction]]

        constraints += [x[InstructionTable.permutation] -
                        processor_instruction_permutation_terminal]

        # ifnewaddress *
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
        #         )
        constraints += [x[InstructionTable.evaluation] -
                        instruction_evaluation_terminal]

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
        num_padded_rows = 0
        for i in range(len(self.matrix)):
            row = self.matrix[i]
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            if new_row[InstructionTable.current_instruction].is_zero():
                num_padded_rows += 1

            # permutation argument
            # update running product?
            # make sure the new row is not padding
            if not new_row[InstructionTable.current_instruction].is_zero():
                # and that the instruction address didn't just change
                if i > 0 and new_row[InstructionTable.address] == xfield.lift(self.matrix[i-1][InstructionTable.address]):
                    permutation_running_product *= alpha - \
                        a * new_row[InstructionTable.address] - \
                        b * new_row[InstructionTable.current_instruction] - \
                        c * new_row[InstructionTable.next_instruction]
                # print("%i:" % i, permutation_running_product)

            new_row += [permutation_running_product]

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
        self.codewords = [[xfield.lift(c) for c in cdwd]
                          for cdwd in self.codewords]

        self.permutation_terminal = permutation_running_product
        self.evaluation_terminal = evaluation_running_sum

