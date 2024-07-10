from table import *
from processor_table import ProcessorTable


class MemoryTable(Table):
    # named indices for base columns
    cycle = 0
    memory_pointer = 1
    memory_value = 2
    dummy = 3

    # named indices for extension columns
    permutation = 4

    def __init__(self, field, length, num_randomizers, generator, order):
        super(MemoryTable, self).__init__(
            field, 4, 5, length, num_randomizers, generator, order)

    # outputs an unpadded but interweaved matrix
    @staticmethod
    def derive_matrix(processor_matrix):
        zero = processor_matrix[0][ProcessorTable.cycle].field.zero()
        one = processor_matrix[0][ProcessorTable.cycle].field.one()

        # copy unpadded rows and sort
        matrix = [[pt[ProcessorTable.cycle], pt[ProcessorTable.memory_pointer],
                   pt[ProcessorTable.memory_value], zero] for pt in processor_matrix if not pt[ProcessorTable.current_instruction].is_zero()]
        matrix.sort(key=lambda mt: mt[MemoryTable.memory_pointer].value)

        # insert dummy rows for smooth clk jumps
        i = 0
        while i < len(matrix)-1:
            if matrix[i][MemoryTable.memory_pointer] == matrix[i+1][MemoryTable.memory_pointer] and matrix[i+1][MemoryTable.cycle] != matrix[i][MemoryTable.cycle] + one:
                matrix.insert(i+1, [matrix[i][MemoryTable.cycle] + one, matrix[i]
                              [MemoryTable.memory_pointer], matrix[i][MemoryTable.memory_value], one])
            i += 1

        return matrix

    def pad(self):
        one = self.matrix[0][MemoryTable.cycle].field.one()
        while len(self.matrix) & (len(self.matrix) - 1) != 0:
            self.matrix.append([self.matrix[-1][MemoryTable.cycle] + one, self.matrix[-1]
                            [MemoryTable.memory_pointer], self.matrix[-1][MemoryTable.memory_value], one])

    @staticmethod
    def transition_constraints_afo_named_variables(cycle, address, value, dummy, cycle_next, address_next, value_next, dummy_next):
        field = list(address.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())

        polynomials = []

        # 1. memory pointer increases by one, zero
        # <=>. (MP*=MP+1) \/ (MP*=MP)
        polynomials += [(address_next - address - one)
                        * (address_next - address)]

        # 2. Only if a) memory pointer does not increase; and b) cycle count increases by one; then the memory value may change
        # a) MV*=/=MV => MP=MP*
        # (DNF:) <=> MV*==MV \/ MP*=MP
        # polynomials += [(value_next-value)*(address_next-address)]
        # b) MV*=/=MV => CLK*=CLK+1
        # (DNF:) <=> MV*==MV \/ CLK*=CLK+1
        # polynomials += [(value_next-value)*(cycle + one - cycle_next)]
        # These constraints are implied by 3.

        # 3. if memory pointer increases by one, then memory value must be set to zero
        #        <=>. MP*=MP+1 => MV* = 0
        # (DNF:) <=>. MP*=/=MP+1 \/ MV*=0
        polynomials += [(address_next - address)
                        * value_next]

        # 4. Dummy has to be zero or one
        # (DNF:) <=> D=1 \/ D=0
        polynomials += [(dummy_next - one) * dummy_next]

        # 5. If Dummy is set, memory pointer cannot change
        #        <=> D=1 => MP*=MP
        # (DNF:) <=> D=/=1 \/ MP*=MP
        polynomials += [dummy * (address_next - address)]

        # 6. If Dummy is set, memory value cannot change
        #        <=> D=1 => MV*=MV
        # (DNF:) <=> D=/=1 \/ MV*=MV
        polynomials += [dummy * (value_next - value)]

        # 7. If the memory pointer remains the same, then the cycle counter has to increase by one
        #        <=> MP*=MP => CLK*=CLK+1
        # (DNF:) <=> MP*=/=MP \/ CLK*=CLK+1
        polynomials += [(address_next - one - address)
                        * (cycle_next - one - cycle)]

        return polynomials

    def base_transition_constraints(self):
        cycle, address, value, dummy, \
            cycle_next, address_next, value_next, dummy_next = MPolynomial.variables(
                2*self.base_width, self.field)
        return MemoryTable.transition_constraints_afo_named_variables(cycle, address, value, dummy, cycle_next, address_next, value_next, dummy_next)

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
        one = MPolynomial.constant(field.one())
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(c) for c in challenges]
        cycle, address, value, dummy, permutation,  \
            cycle_next, address_next, value_next, dummy_next, permutation_next = MPolynomial.variables(
                2*self.full_width, field)

        polynomials = MemoryTable.transition_constraints_afo_named_variables(
            cycle, address, value, dummy, cycle_next, address_next, value_next, dummy_next)

        polynomials += [(permutation *
                        (beta - d * cycle
                         - e * address
                         - f * value)
                        - permutation_next) * (one - dummy) + (permutation - permutation_next) * dummy]

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
        one = MPolynomial.constant(field.one())
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(ch) for ch in challenges]
        permutation = terminals[1]
        x = MPolynomial.variables(self.full_width, field)

        # [permutation *
        #                 (beta - d * cycle
        #                  - e * address
        #                  - f * value)
        #                 - permutation_next]

        # [(permutation *
        #             (beta - d * cycle
        #              - e * address
        #              - f * value)
        #             - permutation_next) * (one - dummy) + (permutation - permutation_next) * dummy]

        return [(x[MemoryTable.permutation] *
                 (beta - d * x[MemoryTable.cycle]
                  - e * x[MemoryTable.memory_pointer]
                  - f * x[MemoryTable.memory_value])
                 - MPolynomial.constant(permutation)) * (one - x[MemoryTable.dummy])
                + (x[MemoryTable.permutation] - MPolynomial.constant(permutation)) * x[MemoryTable.dummy]]

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

            extended_matrix += [new_row]

            if new_row[MemoryTable.dummy].is_zero():
                memory_permutation_running_product *= beta \
                    - d * new_row[MemoryTable.cycle] \
                    - e * new_row[MemoryTable.memory_pointer] \
                    - f * new_row[MemoryTable.memory_value]

        self.matrix = extended_matrix
        self.field = xfield
        self.codewords = [[xfield.lift(c) for c in cdwd]
                          for cdwd in self.codewords]
        self.permutation_terminal = memory_permutation_running_product
