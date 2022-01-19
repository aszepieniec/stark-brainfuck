from io_table import *
from table_extension import TableExtension


class IOExtension(TableExtension):
    # name columns
    column = 0
    evaluation = 1

    def __init__(self, gamma):
        super(IOExtension, self).__init__(gamma.field, 1, 2)

        # names for challenges
        self.gamma = MPolynomial.constant(gamma)
        self.challenges = [gamma]

        self.width = 1+1

        self.evaluation_terminal = self.field.zero()  # initial value

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

        # loop over all rows of table
        for i in range(len(io_table.table)):
            row = io_table.table[i]
            new_row = [xfield.lift(nr) for nr in row]

            new_row += [io_running_evaluation]

            io_running_evaluation = io_running_evaluation * \
                gamma + new_row[0]

            table_extension += [new_row]

        extended_io_table = IOExtension(gamma)
        extended_io_table.table = table_extension
        extended_io_table.evaluation_terminal = io_running_evaluation
        extended_io_table.terminals = [io_running_evaluation]

        return extended_io_table

    def transition_constraints_ext(self, challenges):
        input_, evaluation, \
            input_next, evaluation_next = MPolynomial.variables(4, self.field)
        gamma = MPolynomial.constant(challenges[0])

        polynomials = []

        polynomials += [evaluation * gamma + input_ - evaluation_next]

        return polynomials

    def boundary_constraints_ext(self):
        # format: (row, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        zero = MPolynomial.zero()
        return [(0, x[1] - zero)]  # evaluation

    def terminal_constraints_ext(self, challenges, terminals):
        gamma = MPolynomial.constant(challenges[0])
        evaluation_terminal = MPolynomial.constant(terminals[0])
        x = MPolynomial.variables(self.width, self.field)

        # polynomials += [evaluation * gamma + input_ - evaluation_next]

        return [x[IOExtension.evaluation] * gamma + x[IOTable.column] - evaluation_terminal]
