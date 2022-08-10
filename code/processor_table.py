from table import *


class ProcessorTable(Table):
    # named indices for base columns (=register)
    cycle = 0
    instruction_pointer = 1
    current_instruction = 2
    next_instruction = 3
    memory_pointer = 4
    memory_value = 5
    memory_value_inverse = 6

    # named indices for extension columns
    instruction_permutation = 7
    memory_permutation = 8
    input_evaluation = 9
    output_evaluation = 10

    def __init__(self, field, length, num_randomizers, generator, order):
        super(ProcessorTable, self).__init__(
            field, 7, 11, length, num_randomizers, generator, order)

    def pad(self):
        while len(self.matrix) & (len(self.matrix)-1) != 0:
            new_row = [self.field.zero()] * 7
            new_row[ProcessorTable.cycle] = self.matrix[-1][ProcessorTable.cycle] + \
                self.field.one()
            new_row[ProcessorTable.instruction_pointer] = self.matrix[-1][ProcessorTable.instruction_pointer]
            new_row[ProcessorTable.current_instruction] = self.field.zero()
            new_row[ProcessorTable.next_instruction] = self.field.zero()
            new_row[ProcessorTable.memory_pointer] = self.matrix[-1][ProcessorTable.memory_pointer]
            new_row[ProcessorTable.memory_value] = self.matrix[-1][ProcessorTable.memory_value]
            new_row[ProcessorTable.memory_value_inverse] = self.matrix[-1][ProcessorTable.memory_value_inverse]
            self.matrix += [new_row]

    @staticmethod
    def if_instruction(instruction, indeterminate: MPolynomial):
        '''if_instruction(instr, X)
        returns a polynomial in X that evaluates to 0 in X=FieldElement(instr)'''
        field = list(indeterminate.dictionary.values())[0].field
        # max degree 1
        return MPolynomial.constant(field(ord(instruction))) - indeterminate

    @staticmethod
    def ifnot_instruction(instruction, indeterminate: MPolynomial):
        '''ifnot_instruction(instr, X)
        returns a polynomial in X that evaluates to 0 in all instructions except for X=FieldElement(instr)'''
        field = list(indeterminate.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())
        acc = one
        for c in "[]<>,.+-":
            if c != instruction:
                acc *= indeterminate - \
                    MPolynomial.constant(field(ord(c)))
        return acc  # max degree: 7

    @staticmethod
    def instruction_polynomials(instr, cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, memory_value_inverse, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, memory_value_inverse_next):
        zero = MPolynomial.zero()
        field = list(cycle.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())
        two = MPolynomial.constant(field.one()+field.one())
        polynomials = [zero] * 3
        memory_value_is_zero = memory_value * memory_value_inverse - one

        if instr == '[':
            polynomials[0] = memory_value * (instruction_pointer_next - instruction_pointer - two) + \
                memory_value_is_zero * \
                (instruction_pointer_next - next_instruction)
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value

        elif instr == ']':
            polynomials[0] = memory_value_is_zero * (instruction_pointer_next - instruction_pointer - two) + \
                memory_value * (instruction_pointer_next - next_instruction)
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value

        elif instr == '<':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - \
                memory_pointer + one
            # memory value, satisfied by permutation argument
            polynomials[2] = zero

        elif instr == '>':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - \
                memory_pointer - one
            # memory value, satisfied by permutation argument
            polynomials[2] = zero

        elif instr == '+':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - \
                memory_value - one

        elif instr == '-':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - \
                memory_value + one

        elif instr == ',':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            # memory value, set by evaluation argument
            polynomials[2] = zero

        elif instr == '.':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value

        # account for padding:
        # deactivate all polynomials if current instruction is zero
        for i in range(len(polynomials)):
            polynomials[i] *= current_instruction

        return polynomials  # max degree: 4

    @staticmethod
    def transition_constraints_afo_named_variables(cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, memory_value_inverse, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, memory_value_inverse_next):
        field = list(cycle.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())

        polynomials = [MPolynomial.zero()] * 3

        # instruction-specific polynomials
        for c in "[]<>+-,.":
            # max deg 4
            instr = ProcessorTable.instruction_polynomials(c,
                                                           cycle,
                                                           instruction_pointer,
                                                           current_instruction,
                                                           next_instruction,
                                                           memory_pointer,
                                                           memory_value,
                                                           memory_value_inverse,
                                                           cycle_next,
                                                           instruction_pointer_next,
                                                           current_instruction_next,
                                                           next_instruction_next,
                                                           memory_pointer_next,
                                                           memory_value_next,
                                                           memory_value_inverse_next)
            # max deg: 7
            deselector = ProcessorTable.ifnot_instruction(
                c, current_instruction)

            for i in range(len(instr)):
                # max deg: 11
                polynomials[i] += deselector * instr[i]

        # instruction-independent polynomials
        polynomials += [cycle_next - cycle - one]  # cycle increases by one

        memory_value_is_zero = memory_value * memory_value_inverse - one
        # Verify that `memory_value_inverse` follows the rules
        polynomials += [memory_value * memory_value_is_zero]
        polynomials += [memory_value_inverse * memory_value_is_zero]

        return polynomials  # max deg 11

    def base_transition_constraints(self):
        cycle, \
            instruction_pointer, \
            current_instruction, \
            next_instruction, \
            memory_pointer, \
            memory_value, \
            memory_value_inverse, \
            cycle_next, \
            instruction_pointer_next, \
            current_instruction_next, \
            next_instruction_next, \
            memory_pointer_next, \
            memory_value_next, \
            memory_value_inverse_next = MPolynomial.variables(14, self.field)

        return ProcessorTable.transition_constraints_afo_named_variables(cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, memory_value_inverse, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, memory_value_inverse_next)

    def base_boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.base_width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        constraints = [x[ProcessorTable.cycle] - zero,
                       x[ProcessorTable.instruction_pointer] - zero,
                       # ???, # current instruction
                       # ???, # next instruction
                       x[ProcessorTable.memory_pointer] - zero,
                       x[ProcessorTable.memory_value] - zero,
                       x[ProcessorTable.memory_value_inverse] - zero]

        return constraints

      #
    # # #
      #

    @staticmethod
    def instruction_zerofier(current_instruction):
        field = list(current_instruction.dictionary.values())[0].field
        acc = MPolynomial.constant(field.one())
        for ch in ['[', ']', '<', '>', '+', '-', ',', '.']:
            acc *= current_instruction - \
                MPolynomial.constant(field(ord(ch)))
        return acc

    def transition_constraints_ext(self, challenges):
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(ch) for ch in challenges]
        field = challenges[0].field

        # names for variables
        cycle, \
            instruction_pointer, \
            current_instruction, \
            next_instruction, \
            memory_pointer, \
            memory_value, \
            memory_value_inverse, \
            instruction_permutation, \
            memory_permutation, \
            input_evaluation, \
            output_evaluation, \
            cycle_next, \
            instruction_pointer_next, \
            current_instruction_next, \
            next_instruction_next, \
            memory_pointer_next, \
            memory_value_next, \
            memory_value_inverse_next, \
            instruction_permutation_next, \
            memory_permutation_next, \
            input_evaluation_next, \
            output_evaluation_next = MPolynomial.variables(22, field)

        # base AIR polynomials
        polynomials = ProcessorTable.transition_constraints_afo_named_variables(cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value,
                                                                                memory_value_inverse, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, memory_value_inverse_next)

        assert (len(polynomials) ==
                6), f"expected to have 6 transition constraint polynomials, but have {len(polynomials)}"

        # extension AIR polynomials
        # running product for instruction permutation
        polynomials += [(instruction_permutation *
                        (alpha - a * instruction_pointer
                         - b * current_instruction
                         - c * next_instruction)
                        - instruction_permutation_next) * current_instruction +
                        self.instruction_zerofier(current_instruction) * (instruction_permutation - instruction_permutation_next)]
        # polynomials += [cycle-cycle] # zero

        # running product for memory permutation
        polynomials += [(memory_permutation *
                        (beta - d * cycle
                         - e * memory_pointer - f * memory_value)
                        - memory_permutation_next) * current_instruction + (memory_permutation - memory_permutation_next) * ProcessorTable.instruction_zerofier(current_instruction)]
        # running evaluation for input
        polynomials += [(input_evaluation_next - input_evaluation * gamma - memory_value_next) * ProcessorTable.ifnot_instruction(
            ',', current_instruction) * current_instruction + (input_evaluation_next - input_evaluation) * ProcessorTable.if_instruction(',', current_instruction)]
        # running evaluation for output
        polynomials += [(output_evaluation_next - output_evaluation * delta - memory_value) * ProcessorTable.ifnot_instruction(
            '.', current_instruction) * current_instruction + (output_evaluation_next - output_evaluation) * ProcessorTable.if_instruction('.', current_instruction)]

        assert (len(polynomials) ==
                10), f"number of transition constraints ({len(polynomials)}) does not match with expectation (10)"

        return polynomials  # max degree 11

    def boundary_constraints_ext(self, challenges):
        field = challenges[0].field
        # format: mpolynomial
        x = MPolynomial.variables(self.full_width, field)
        one = MPolynomial.constant(field.one())
        zero = MPolynomial.zero()
        constraints = [x[self.cycle] - zero,
                       x[self.instruction_pointer] - zero,
                       # x[self.current_instruction] - ??),
                       # x[self.next_instruction] - ??),
                       x[self.memory_pointer] - zero,
                       x[self.memory_value] - zero,
                       x[self.memory_value_inverse] - zero,
                       # x[self.instruction_permutation] - one,
                       # x[self.memory_permutation] - one,
                       x[self.input_evaluation] - zero,
                       x[self.output_evaluation] - zero
                       ]
        assert (len(constraints) ==
                7), "number of boundary constraints does not match with expectation"
        return constraints

    def terminal_constraints_ext(self, challenges, terminals):
        field = challenges[0].field
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [
            MPolynomial.constant(ch) for ch in challenges]
        x = MPolynomial.variables(self.full_width, field)
        airs = []

        # running product for instruction permutation
        # polynomials += [(instruction_permutation *
        #                 (self.alpha
        #                   - self.a * instruction_pointer
        #                   - self.b * current_instruction
        #                   - self.c * next_instruction)
        #                 - instruction_permutation_next) * current_instruction]
        airs += [MPolynomial.constant(terminals[0]) -
                 x[ProcessorTable.instruction_permutation]]

        # running product for memory permutation
        # polynomials += [(memory_permutation *
        #                   (beta
        #                       - d * cycle
        #                       - e * memory_pointer
        #                       - f * memory_value)
        #                   - memory_permutation_next) * current_instruction
        #               + (memory_permutation - memory_permutation_next)
        #                   * ProcessorTable.instruction_zerofier(current_instruction)]
        airs += [(MPolynomial.constant(terminals[1])
                  - x[ProcessorTable.memory_permutation]
                  * (beta
                     - d * x[ProcessorTable.cycle]
                     - e * x[ProcessorTable.memory_pointer]
                     - f * x[ProcessorTable.memory_value]))
                 * x[ProcessorTable.current_instruction]
                 + (MPolynomial.constant(terminals[1]) -
                    x[ProcessorTable.memory_permutation])
                 * ProcessorTable.instruction_zerofier(x[ProcessorTable.current_instruction])]

        # running evaluation for input
        # polynomials += [(input_evaluation_next \
        #                   - input_evaluation * self.gamma \
        #                   - memory_value) * ProcessorTable.ifnot_instruction(',', current_instruction) * current_instruction \
        #               + (input_evaluation_next - input_evaluation) * ProcessorTable.if_instruction(',', current_instruction)]
        airs += [MPolynomial.constant(terminals[2]) -
                 x[ProcessorTable.input_evaluation]]

        # running evaluation for output
        # polynomials += [(output_evaluation_next - output_evaluation * self.delta - memory_value) * ProcessorTable.ifnot_instruction(
        #     '.', current_instruction) * current_instruction + (output_evaluation_next - output_evaluation) * ProcessorTable.if_instruction('.', current_instruction)]
        airs += [MPolynomial.constant(terminals[3]) -
                 x[ProcessorTable.output_evaluation]]

        assert (len(airs) ==
                4), "number of terminal airs did not match with expectation"
        return airs

    def extend(self, all_challenges, all_initials):
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = all_challenges
        processor_instruction_permutation_initial, processor_memory_permutation_initial = all_initials

        # algebra stuff
        field = self.field
        xfield = a.field
        one = xfield.one()
        zero = xfield.zero()

        # prepare for loop
        instruction_permutation_running_product = processor_instruction_permutation_initial
        memory_permutation_running_product = processor_memory_permutation_initial
        input_evaluation_running_evaluation = zero
        output_evaluation_running_evaluation = zero

        # loop over all rows
        extended_matrix = []
        for i in range(len(self.matrix)):
            row = self.matrix[i]

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # next, define the additional columns

            # 1. running product for instruction permutation
            new_row += [instruction_permutation_running_product]
            # if not padding
            if not new_row[ProcessorTable.current_instruction].is_zero():
                instruction_permutation_running_product *= alpha - \
                    a * new_row[ProcessorTable.instruction_pointer] - \
                    b * new_row[ProcessorTable.current_instruction] - \
                    c * new_row[ProcessorTable.next_instruction]
                # print("%i." % i, instruction_permutation_running_product)

            # 2. running product for memory access
            new_row += [memory_permutation_running_product]
            if not new_row[ProcessorTable.current_instruction].is_zero():
                memory_permutation_running_product *= beta \
                    - d * new_row[ProcessorTable.cycle] \
                    - e * new_row[ProcessorTable.memory_pointer] \
                    - f * new_row[ProcessorTable.memory_value]

            # 3. evaluation for input
            new_row += [input_evaluation_running_evaluation]
            if row[ProcessorTable.current_instruction] == BaseFieldElement(ord(','), field):
                input_evaluation_running_evaluation = input_evaluation_running_evaluation * gamma \
                    + xfield.lift(self.matrix[i+1]
                                  [ProcessorTable.memory_value])
                # the memory-value register only assumes the input value after the instruction has been performed

            # 4. evaluation for output
            new_row += [output_evaluation_running_evaluation]
            if row[ProcessorTable.current_instruction] == BaseFieldElement(ord('.'), field):
                output_evaluation_running_evaluation = output_evaluation_running_evaluation * delta \
                    + new_row[ProcessorTable.memory_value]

            extended_matrix += [new_row]

        self.field = xfield
        self.matrix = extended_matrix
        self.codewords = [[xfield.lift(c) for c in cdwd]
                          for cdwd in self.codewords]

        self.instruction_permutation_terminal = instruction_permutation_running_product
        self.memory_permutation_terminal = memory_permutation_running_product
        self.input_evaluation_terminal = input_evaluation_running_evaluation
        self.output_evaluation_terminal = output_evaluation_running_evaluation
