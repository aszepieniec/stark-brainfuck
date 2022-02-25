from io_table import *
from table_extension import TableExtension


class IOExtension(TableExtension):
    # name columns
    column = 0
    evaluation = 1

    width = 2

    def __init__(self, length, generator, order, gamma, evaluation_terminal):

        num_randomizers = 0
        super(IOExtension, self).__init__(
            gamma.field, IOTable.width, IOExtension.width, length, num_randomizers, generator, order)

        # length is before, height is after, rounding to the next power of two
        self.length = length

        # names for challenges
        self.gamma = MPolynomial.constant(gamma)
        self.challenges = [gamma]

        self.evaluation_terminal = evaluation_terminal
        self.terminals = [evaluation_terminal]

    @staticmethod
    def prepare_verify(log_num_rows, challenges, terminals):
        io_extension = IOExtension(challenges[0])
        io_extension.evaluation_terminal = terminals[0]
        io_extension.log_num_rows = log_num_rows
        io_extension.terminals = terminals
        return io_extension

    @staticmethod
    def extend(io_table, gamma):

        # algebra stuff
        xfield = gamma.field
        zero = xfield.zero()
        one = xfield.one()

        # prepare loop
        table_extension = []
        io_running_evaluation = zero
        evaluation_terminal = zero

        # loop over all rows of table
        for i in range(len(io_table.table)):
            row = io_table.table[i]
            new_row = [xfield.lift(nr) for nr in row]

            new_row += [io_running_evaluation]

            io_running_evaluation = io_running_evaluation * \
                gamma + new_row[0]

            if i == io_table.length - 1:
                evaluation_terminal = io_running_evaluation

            table_extension += [new_row]

        assert(io_table.height & (io_table.height - 1)
               == 0), f"height of io_table must be 2^k"
        extended_io_table = IOExtension(
            io_table.length, io_table.generator, io_table.order, gamma, evaluation_terminal)
        extended_io_table.table = table_extension

        extended_io_table.field = xfield

        return extended_io_table

    def transition_constraints_ext(self, challenges):
        input_, evaluation, \
            input_next, evaluation_next = MPolynomial.variables(
                2*IOExtension.width, self.field)
        gamma = MPolynomial.constant(challenges[0])

        polynomials = []

        polynomials += [evaluation * gamma + input_ - evaluation_next]

        return polynomials

    def boundary_constraints_ext(self):
        # format: mpolynomial
        x = MPolynomial.variables(self.width, self.field)
        zero = MPolynomial.zero()
        return [x[IOExtension.evaluation] - zero]  # evaluation

    def terminal_constraints_ext(self, challenges, terminals):
        if self.height == 0:
            assert(terminals[0].is_zero(
            )), "evaluation terminal for IOExtension has to be zero when the table has zero rows"

        if self.height != 0:
            assert(not terminals[0].is_zero(
            )), "evaluation terminal for non-empty IOExtension is zero but shouldn't be!"

        gamma = challenges[0]
        gamma_mpoly = MPolynomial.constant(gamma)

        evaluation_terminal = MPolynomial.constant(terminals[0])
        x = MPolynomial.variables(self.width, self.field)

        # `evaluation_terminal` is the value of the running sum variable
        # after the `self.length`th row. In every additional row, it is
        # multiplied by another factor gamma. So we multiply by gamma^diff
        # to get the value of the evaluation terminal after all 2^k rows.
        actual_terminal = evaluation_terminal * \
            MPolynomial.constant(gamma ^ (self.height - self.length))

        print("in IOExtension -- actual terminal:", actual_terminal, type(actual_terminal), "but evaluation terminal:",
              evaluation_terminal, "offset is gamma^", self.height - self.length, "=", self.height, "-", self.length)

        # polynomials += [evaluation * gamma + input_ - evaluation_next]

        return [x[IOExtension.evaluation] * gamma_mpoly + x[IOTable.column] - actual_terminal]
