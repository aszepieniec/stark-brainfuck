from processor_table import *

class ProcessorExtension(ProcessorTable):
    def __init__(self, a, b, c, d, e, f, alpha, beta, gamma, delta):
        field = a.field

        # names for challenges
        self.a = MPolynomial.constant(a)
        self.b = MPolynomial.constant(b)
        self.c = MPolynomial.constant(c)
        self.d = MPolynomial.constant(d)
        self.e = MPolynomial.constant(e)
        self.f = MPolynomial.constant(f)
        self.alpha = MPolynomial.constant(alpha)
        self.beta = MPolynomial.constant(beta)
        self.gamma = MPolynomial.constant(gamma)
        self.delta = MPolynomial.constant(delta)

        super(ProcessorExtension, self).__init__(field)
        self.width = 7 + 6

    @staticmethod
    def extend( processor_table, a, b, c, d, e, f, alpha, beta, gamma, delta):
        # register names
        cycle = 0
        instruction_pointer = 1
        current_instruction = 2
        next_instruction = 3
        memory_pointer = 4
        memory_value = 5
        is_zero = 6

        # algebra stuff
        field = processor_table.field
        xfield = a.field
        one = xfield.one()
        zero = xfield.zero()

        # prepare for loop
        instruction_permutation_running_product = one
        memory_permutation_running_product = one
        input_evaluation = zero
        input_indeterminate = one
        output_evaluation = zero
        output_indeterminate = one

        # loop over all rows
        table_extension = []
        for row in processor_table.table:
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # next, define the additional columns

            # 1. running product for instruction permutation
            new_row += [instruction_permutation_running_product]
            instruction_permutation_running_product *= alpha - a * \
                new_row[instruction_pointer] - b * \
                new_row[current_instruction] - c * new_row[next_instruction]

            # 2. running product for memory access
            memory_permutation_running_product *= beta - d * \
                new_row[cycle] - e * new_row[memory_pointer] - f * new_row[memory_value]
            new_row += [memory_permutation_running_product]

            # 3. evaluation for input
            new_row += [input_indeterminate]
            new_row += [input_evaluation]
            if row[current_instruction] == BaseFieldElement(ord(','), field):
                input_indeterminate *= gamma
                input_evaluation += input_indeterminate * new_row[memory_value]

            # 4. evaluation for output
            new_row += [output_indeterminate]
            new_row += [output_evaluation]
            if row[current_instruction] == BaseFieldElement(ord('.'), field):
                output_evaluation += output_indeterminate * new_row[memory_value]
                output_indeterminate *= delta

            table_extension += [new_row]

        extended_processor_table = ProcessorExtension(a, b, c, d, e, f, alpha, beta, gamma, delta)
        extended_processor_table.table = table_extension

        return extended_processor_table

    def transition_constraints(self):
        # names for variables
        cycle, \
            instruction_pointer, \
            current_instruction, \
            next_instruction, \
            memory_pointer, \
            memory_value, \
            is_zero, \
            instruction_permutation, \
            memory_permutation, \
            input_indeterminate, \
            input_evaluation, \
            output_indeterminate, \
            output_evaluation, \
            cycle_next, \
            instruction_pointer_next, \
            current_instruction_next, \
            next_instruction_next, \
            memory_pointer_next, \
            memory_value_next, \
            is_zero_next, \
            instruction_permutation_next, \
            memory_permutation_next, \
            input_indeterminate_next, \
            input_evaluation_next, \
            output_indeterminate_next, \
            output_evaluation_next = MPolynomial.variables(26, self.field)

        # base AIR polynomials
        polynomials = self.transition_constraints_afo_named_variables(cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, is_zero, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, is_zero_next)

        # extension AIR polynomials
        # running product for instruction permutation
        polynomials += [instruction_permutation * \
                            ( self.alpha - self.a * instruction_pointer \
                                - self.b * current_instruction \
                                - self.c * next_instruction ) \
                             - instruction_permutation_next]
        # running product for memory permutation
        polynomials += [memory_permutation * \
                            ( self.beta - self.d * cycle_next \
                                - self.e * memory_pointer_next  - self.f * memory_value_next ) \
                             - memory_permutation_next]
        # running evaluation for input 
        polynomials += [(input_indeterminate_next - input_indeterminate * self.gamma) * self.ifnot_instruction(',', current_instruction) + (input_indeterminate_next - input_indeterminate) * self.if_instruction(',', current_instruction)]
        polynomials += [(input_evaluation_next - input_evaluation - input_indeterminate * memory_value) * self.ifnot_instruction(',', current_instruction) + (input_evaluation_next - input_evaluation) * self.if_instruction(',', current_instruction)]
        # running evaluation for output
        polynomials += [(output_indeterminate_next - output_indeterminate * self.delta) * self.ifnot_instruction('.', current_instruction) + (output_indeterminate_next - output_indeterminate) * self.if_instruction('.', current_instruction)]
        polynomials += [(output_evaluation_next - output_evaluation - output_indeterminate * memory_value) * self.ifnot_instruction('.', current_instruction) + (output_evaluation_next - output_evaluation) * self.if_instruction('.', current_instruction)]

        return polynomials

    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        constraints = [(0, x[0] - zero),  # cycle
                       # instruction pointer
                       (0, x[1] - zero),
                       # (0, x[2] - ??), # current instruction
                       # (0, x[3] - ??), # next instruction
                       (0, x[4] - zero),  # memory pointer
                       (0, x[5] - zero),  # memory value
                       (0, x[6] - one),   # memval==0
                       (0, x[7] - one),   # running product for instruction permutation
                       (0, x[8] - self.beta + self.d * x[0] + self.e * x[4] + self.f * x[5]),   # running product for memory permutation
                       (0, x[9] - one),   # running power for input
                       (0, x[10] - zero), # running evaluation for input
                       (0, x[11] - one),  # running power for output
                       (0, x[12] - zero)  # running evaluation for output
                       ]
        return constraints

    def instruction_terminal( self ):
    def memory_terminal( self ):
    def input_terminal( self ):
    def output_terminal( self ):