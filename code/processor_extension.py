from processor_table import *

class ProcessorExtension(ProcessorTable):
    def __init__(self, challenges):
        field = challenges[0].field

        # names for challenges
        challenges = [MPolynomial.constant(c) for c in challenges]
        self.a = challenges[0]
        self.b = challenges[1]
        self.c = challenges[2]
        self.d = challenges[3]
        self.e = challenges[4]
        self.f = challenges[5]
        self.alpha = challenges[6]
        self.beta = challenges[7]
        self.gamma = challenges[8]
        self.delta = challenges[9]

        super(ProcessorExtension, self).__init__(field)
        self.width = 7 + 6

    def transition_constraints(self, challenges):
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
                            ( self.beta - self.d * memory_pointer_next \
                                - self.e * memory_value_next ) \
                             - memory_permutation_next]
        # running evaluation for input 
        polynomials += [input_indeterminate_next - input_indeterminate * self.gamma * self.if_instruction(',', current_instruction) - input_indeterminate * self.ifnot_instruction(',', current_instruction)]
        polynomials += [input_evaluation + input_indeterminate * memory_value * self.if_instruction(',', current_instruction) - input_evaluation_next]
        # running evaluation for output
        polynomials += [output_indeterminate_next - output_indeterminate * self.delta * self.if_instruction('.', current_instruction) - output_indeterminate * self.ifnot_instruction('.', current_instruction)]
        polynomials += [output_evaluation + output_indeterminate * memory_value * self.if_instruction('.', current_instruction) - output_evaluation_next]

        return polynomials

    def boundary_constraints(self):
        # format: (register, cycle, value)
        constraints = [(0, 0, self.field.zero()),  # cycle
                       # instruction pointer
                       (1, 0, self.field.zero()),
                       # (2, 0, ???), # current instruction
                       # (3, 0, ???), # next instruction
                       (4, 0, self.field.zero()),  # memory pointer
                       (5, 0, self.field.zero()),  # memory value
                       (6, 0, self.field.one()),   # memval==0
                       (7, 0, self.field.one()),   # running product for instruction permutation
                       (8, 0, self.field.one()),   # running product for memory permutation
                       (9, 0, self.field.one()),   # running power for input
                       (10, 0, self.field.zero()), # running evaluation for input
                       (11, 0, self.field.one()),  # running power for output
                       (12, 0, self.field.zero())  # running evaluation for output
                       ]
        return constraints
