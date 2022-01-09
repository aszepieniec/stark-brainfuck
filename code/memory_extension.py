from memory_table import *


class MemoryExtension(MemoryTable):
    def __init__(self, d, e, f, beta):
        field = d.field

        cycle = 0
        memory_pointer = 1
        memory_value = 2
        permutation = 3

        # terminal values (placeholder)
        self.permutation_terminal = field.zero()

        self.d = MPolynomial.constant(d)
        self.e = MPolynomial.constant(e)
        self.f = MPolynomial.constant(f)
        self.beta = MPolynomial.constant(beta)

        super(MemoryTable, self).__init__(field, 3+1)

    @staticmethod
    def extend(memory_table, d, e, f, beta):

        # algebra stuff
        field = memory_table.field
        xfield = d.field
        one = xfield.one()

        # prepare loop
        table_extension = []
        memory_permutation_running_product = one

        # loop over all rows of table
        for row in memory_table.table:
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # match with this:
            # 2. running product for memory access

            new_row += [memory_permutation_running_product]
            memory_permutation_running_product *= beta - \
                d * new_row[0] - e * new_row[1] - f * new_row[2]

            table_extension += [new_row]

        extended_memory_table = MemoryExtension(d, e, f, beta)
        extended_memory_table.table = table_extension

        extended_memory_table.permutation_terminal = memory_permutation_running_product

        return extended_memory_table

    def transition_constraints(self):
        cycle, address, value, permutation, \
            cycle_next, address_next, value_next, permutation_next = MPolynomial.variables(
                8, self.field)

        polynomials = MemoryTable.transition_constraints_afo_named_variables(
            cycle, address, value, cycle_next, address_next, value_next)

        polynomials += [permutation *
                        (self.beta - self.d * cycle
                         - self.e * address
                         - self.f * value)
                        - permutation_next]

        return polynomials

    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[0] - zero),  # cycle
                (0, x[1] - zero),  # memory pointer
                (0, x[2] - zero),  # memory value
                (0, x[3] - one),   # permutation
                ]
