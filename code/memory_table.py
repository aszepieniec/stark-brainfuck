from table import *
from processor_table import ProcessorTable


class MemoryTable(Table):
    # named indices for base columns
    cycle = 0
    memory_pointer = 1
    memory_value = 2

    # named indices for extension columns
    permutation = 3

    def __init__(self, field, length, num_randomizers, generator, order):
        super(MemoryTable, self).__init__(
            field, 3, 4, length, num_randomizers, generator, order)

    @staticmethod
    def derive_matrix(processor_matrix, num_randomizers):
        matrix = [[pt[ProcessorTable.cycle], pt[ProcessorTable.memory_pointer],
                  pt[ProcessorTable.memory_value]] for pt in processor_matrix]
        matrix.sort(key=lambda mt: mt[MemoryTable.memory_pointer].value)
        return matrix

    @staticmethod
    def transition_constraints_afo_named_variables(cycle, address, value, cycle_next, address_next, value_next):
        field = list(address.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())

        polynomials = []

        # 1. memory pointer increases by one, zero, or minus one
        # <=>. (MP*=MP+1) \/ (MP*=MP)
        polynomials += [(address_next - address - one)
                        * (address_next - address)]

        # 2. if memory pointer does not increase, then memory value can change only if cycle counter increases by one
        #        <=>. MP*=MP => (MV*=/=MV => CLK*=CLK+1)
        #        <=>. MP*=/=MP \/ (MV*=/=MV => CLK*=CLK+1)
        # (DNF:) <=>. MP*=/=MP \/ MV*=MV \/ CLK*=CLK+1
        polynomials += [(address_next - address - one) *
                        (value_next - value) * (cycle_next - cycle - one)]

        # 3. if memory pointer increases by one, then memory value must be set to zero
        #        <=>. MP*=MP+1 => MV* = 0
        # (DNF:) <=>. MP*=/=MP+1 \/ MV*=0
        polynomials += [(address_next - address)
                        * value_next]

        return polynomials

    def base_transition_constraints(self):
        cycle, address, value, \
            cycle_next, address_next, value_next = MPolynomial.variables(
                6, self.field)
        return MemoryTable.transition_constraints_afo_named_variables(cycle, address, value, cycle_next, address_next, value_next)

    def base_boundary_constraints(self):
        # format: mpolynomial
        x = MPolynomial.variables(self.base_width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [x[MemoryTable.cycle],
                x[MemoryTable.memory_pointer],
                x[MemoryTable.memory_value],
                ]

      #
    # # #
      #

    def transition_constraints_ext(self, challenges):
        field = challenges[0].field
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(c) for c in challenges]
        cycle, address, value, permutation, \
            cycle_next, address_next, value_next, permutation_next = MPolynomial.variables(
                8, field)

        polynomials = MemoryTable.transition_constraints_afo_named_variables(
            cycle, address, value, cycle_next, address_next, value_next)

        assert(len(polynomials) ==
               3), f"number of transition constraints from MemoryTable is {len(polynomials)}, but expected 3"

        polynomials += [permutation *
                        (beta - d * cycle
                         - e * address
                         - f * value)
                        - permutation_next]

        return polynomials

    def boundary_constraints_ext(self, challenges):
        field = challenges[0].field
        # format: mpolynomial
        x = MPolynomial.variables(self.full_width, field)
        one = MPolynomial.constant(field.one())
        zero = MPolynomial.zero()
        return [x[MemoryTable.cycle] - zero,  # cycle
                x[MemoryTable.memory_pointer] - zero,  # memory pointer
                x[MemoryTable.memory_value] - zero,  # memory value
                # x[MemoryExtension.permutation] - one   # permutation
                ]

    def terminal_constraints_ext(self, challenges, terminals):
        field = challenges[0].field
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(ch) for ch in challenges]
        permutation = terminals[1]
        x = MPolynomial.variables(self.full_width, field)

        # [permutation *
        #                 (beta - d * cycle
        #                  - e * address
        #                  - f * value)
        #                 - permutation_next]

        return [x[MemoryTable.permutation] * (beta - d * x[MemoryTable.cycle] - e * x[MemoryTable.memory_pointer] - f * x[MemoryTable.memory_value]) - MPolynomial.constant(permutation)]

    def extend(self, all_challenges, all_initials):
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = all_challenges
        processor_instruction_permutation_initial, processor_memory_permutation_initial = all_initials

        # algebra stuff
        field = self.field
        xfield = d.field
        one = xfield.one()

        # prepare loop
        extended_matrix = []
        memory_permutation_running_product = processor_memory_permutation_initial

        # loop over all rows of table
        for i in range(len(self.matrix)):
            row = self.matrix[i]
            new_row = [xfield.lift(nr) for nr in row]

            new_row += [memory_permutation_running_product]
            memory_permutation_running_product *= beta \
                - d * new_row[MemoryTable.cycle] \
                - e * new_row[MemoryTable.memory_pointer] \
                - f * new_row[MemoryTable.memory_value]

            extended_matrix += [new_row]

        self.matrix = extended_matrix
        self.field = xfield
        self.codewords = [[xfield.lift(c) for c in cdwd]
                          for cdwd in self.codewords]
        self.permutation_terminal = memory_permutation_running_product
