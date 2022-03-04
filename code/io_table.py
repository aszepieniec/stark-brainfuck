from table import *


class IOTable(Table):
    # named indices for base columns
    column = 0

    # named indices for extension columns
    evaluation = 1

    def __init__(self, field, length, generator, order):
        num_randomizers = 0
        super(IOTable, self).__init__(
            field, 1, 2, length, num_randomizers, generator, order)

    def pad(self):
        self.length = len(self.matrix)
        while len(self.matrix) & (len(self.matrix) - 1) != 0:
            self.matrix += [[self.field.zero()]]
        self.height = len(self.matrix)

    def base_transition_constraints(self):
        return []

    def base_boundary_constraints(self):
        return []

      #
    # # #
      #

    def transition_constraints_ext(self, challenges):
        field = challenges[0].field
        input_, evaluation, \
            input_next, evaluation_next = MPolynomial.variables(
                2*self.full_width, field)
        iota = MPolynomial.constant(challenges[self.challenge_index])

        polynomials = []

        polynomials += [evaluation * iota + input_next - evaluation_next]

        return polynomials

    def boundary_constraints_ext(self, challenges):
        field = challenges[0].field
        # format: mpolynomial
        x = MPolynomial.variables(self.full_width, field)
        zero = MPolynomial.zero()
        return [x[IOTable.evaluation] - x[IOTable.column]]  # evaluation

    def terminal_constraints_ext(self, challenges, terminals):

        if self.height != 0:
            assert(not terminals[self.terminal_index].is_zero(
            )), "evaluation terminal for non-empty IOTable is zero but shouldn't be!"

        field = challenges[0].field
        iota = challenges[self.challenge_index]
        offset = MPolynomial.constant(
            iota ^ (self.height - self.length))

        evaluation_terminal = MPolynomial.constant(
            terminals[self.terminal_index])
        x = MPolynomial.variables(self.full_width, field)

        # In every additional row, the running evaluation variable is
        # multiplied by another factor iota. So we multiply by iota^diff
        # to get the value of the evaluation terminal after all 2^k rows.
        actual_terminal = evaluation_terminal * offset

        print("terminal index:", self.terminal_index)
        print("in IOTable -- actual terminal:", actual_terminal)
        print("but evaluation terminal:", evaluation_terminal)
        print("offset is iota^", self.height - self.length, " = iota^(",
              self.height, "-", self.length, ") = ", iota ^ (self.height - self.length))
        print("iota:", iota)

        # polynomials += [evaluation * gamma + input_ - evaluation_next]

        return [x[IOTable.evaluation] - evaluation_terminal * offset]

    def extend_iotable(self, iota):

        # algebra stuff
        xfield = iota.field
        zero = xfield.zero()
        one = xfield.one()

        # prepare loop
        extended_matrix = []
        io_running_evaluation = zero
        evaluation_terminal = zero

        print("In ", str(type(self)), "iota is:", iota)

        # loop over all rows of table
        for i in range(len(self.matrix)):
            row = self.matrix[i]
            new_row = [xfield.lift(nr) for nr in row]

            io_running_evaluation = io_running_evaluation * \
                iota + new_row[IOTable.column]
            new_row += [io_running_evaluation]

            if i == self.length - 1:
                evaluation_terminal = io_running_evaluation

            extended_matrix += [new_row]

        assert(self.height & (self.height - 1)
               == 0), f"height of io_table must be 2^k"

        self.field = xfield
        self.matrix = extended_matrix
        self.codewords = [[xfield.lift(c) for c in cdwd]
                          for cdwd in self.codewords]
        self.evaluation_terminal = evaluation_terminal


class InputTable(IOTable):
    def __init__(self, field, length, generator, order):
        super(InputTable, self).__init__(field, length, generator, order)
        self.challenge_index = 8
        self.terminal_index = 2

    def extend(self, all_challenges, all_initials):
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = all_challenges
        processor_instruction_permutation_initial, processor_memory_permutation_initial = all_initials
        self.extend_iotable(all_challenges[self.challenge_index])


class OutputTable(IOTable):
    def __init__(self, field, length, generator, order):
        super(OutputTable, self).__init__(field, length, generator, order)
        self.challenge_index = 9
        self.terminal_index = 3

    def extend(self, all_challenges, all_initials):
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = all_challenges
        processor_instruction_permutation_initial, processor_memory_permutation_initial = all_initials
        self.extend_iotable(all_challenges[self.challenge_index])
