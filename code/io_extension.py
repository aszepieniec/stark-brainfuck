from io_table import *


class IOExtension(IOTable):
    def __init__(self, gamma):
        field = gamma.field

        # names for challenges
        self.gamma = MPolynomial.constant(gamma)

        super(IOExtension, self).__init__(field)
        self.width = 1+1

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

        return extended_io_table

    def transition_constraints(self):
        input_, evaluation, \
            input_next, evaluation_next = MPolynomial.variables(4, self.field)

        polynomials = []

        polynomials += [evaluation * self.gamma + input_ - evaluation_next]

        return polynomials

    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        zero = MPolynomial.zero()
        return [(0, x[1] - zero)]  # evaluation
