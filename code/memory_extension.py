from re import M, T
from memory_table import *
from table_extension import TableExtension


class MemoryExtension(TableExtension):
    # name columns
    cycle = 0
    memory_pointer = 1
    memory_value = 2

    permutation = 3

    width = 4

    def __init__(self, d, e, f, beta):
        super(MemoryExtension, self).__init__(d.field, 3, 4)

        field = d.field

        # terminal values (placeholder)
        self.permutation_terminal = field.zero()

        self.d = MPolynomial.constant(d)
        self.e = MPolynomial.constant(e)
        self.f = MPolynomial.constant(f)
        self.beta = MPolynomial.constant(beta)
        self.challenges = [d, e, f, beta]

    @staticmethod
    def prepare_verify(log_num_rows, challenges, terminals):
        d, e, f, beta = challenges
        memory_extension = MemoryExtension(d, e, f, beta)
        memory_extension.permutation_terminal = terminals[0]
        memory_extension.log_num_rows = log_num_rows
        memory_extension.terminals = terminals
        return memory_extension

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
        for i in range(len(memory_table.table)):
            row = memory_table.table[i]
            new_row = [xfield.lift(nr) for nr in row]

            new_row += [memory_permutation_running_product]
            memory_permutation_running_product *= beta \
                - d * new_row[MemoryExtension.cycle] \
                - e * new_row[MemoryExtension.memory_pointer] \
                - f * new_row[MemoryExtension.memory_value]

            table_extension += [new_row]

        extended_memory_table = MemoryExtension(d, e, f, beta)
        extended_memory_table.table = table_extension

        extended_memory_table.permutation_terminal = memory_permutation_running_product
        extended_memory_table.terminals = [memory_permutation_running_product]
        extended_memory_table.field = xfield

        return extended_memory_table

    def transition_constraints_ext(self, challenges):
        d, e, f, beta = [MPolynomial.constant(c) for c in challenges]
        cycle, address, value, permutation, \
            cycle_next, address_next, value_next, permutation_next = MPolynomial.variables(
                8, self.field)

        polynomials = MemoryTable.transition_constraints_afo_named_variables(
            cycle, address, value, cycle_next, address_next, value_next)

        polynomials += [permutation *
                        (beta - d * cycle
                         - e * address
                         - f * value)
                        - permutation_next]

        return polynomials

    def boundary_constraints_ext(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[MemoryExtension.cycle] - zero),  # cycle
                (0, x[MemoryExtension.memory_pointer] - zero),  # memory pointer
                (0, x[MemoryExtension.memory_value] - zero),  # memory value
                (0, x[MemoryExtension.permutation] - one),   # permutation
                ]

    def interpolate_extension(self, omega, order, num_randomizers):
        return self.interpolate_columns(omega, order, num_randomizers, range(MemoryTable.memory_value, self.width))

    def terminal_constraints_ext(self, challenges, terminals):
        d, e, f, beta = [MPolynomial.constant(c) for c in challenges]
        permutation = terminals[0]
        x = MPolynomial.variables(self.width, self.field)

        # [permutation *
        #                 (beta - d * cycle
        #                  - e * address
        #                  - f * value)
        #                 - permutation_next]

        return [x[MemoryExtension.permutation] * (beta - d * x[MemoryTable.cycle] - e * x[MemoryTable.memory_pointer] - f * x[MemoryTable.memory_value]) - MPolynomial.constant(permutation)]
