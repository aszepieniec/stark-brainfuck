from algebra import *
from input_table import InputTable
from instruction_table import InstructionTable
from memory_table import MemoryTable
from multivariate import *
import sys
from output_table import OutputTable

from processor_table import ProcessorTable


class Register:
    field = BaseField.main()

    def __init__(self):
        self.cycle = Register.field.zero()
        self.instruction_pointer = Register.field.zero()
        self.current_instruction = Register.field.zero()
        self.next_instruction = Register.field.zero()
        self.memory_pointer = Register.field.zero()
        self.memory_value = Register.field.zero()
        self.is_zero = Register.field.one()


class VirtualMachine:
    field = BaseField.main()

    def execute(brainfuck_code):
        program = VirtualMachine.compile(brainfuck_code)
        input_data, output_data = VirtualMachine.perform(program)
        return input_data, output_data

    def compile(brainfuck_code):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        def F(x): return BaseFieldElement(ord(x), field)

        # parser
        program = []
        stack = []
        for symbol in brainfuck_code:
            program += [F(symbol)]
            if symbol == '[':
                program += [zero]
                stack += [len(program)-1]
            elif symbol == ']':
                program += [BaseFieldElement(stack[-1]+1, field)]
                program[stack[-1]] = BaseFieldElement(len(program)+1, field)
                stack = stack[:-1]
        return program

    def perform(program, input_data=None):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        def F(x): return BaseFieldElement(ord(x), field)

        # initial state
        instruction_pointer = 0
        memory_pointer = BaseFieldElement(0, VirtualMachine.field)
        memory = dict()  # field elements to field elements
        output_data = []
        input_counter = 0

        # main loop
        while instruction_pointer < len(program):
            if program[instruction_pointer] == F('['):
                if memory.get(memory_pointer, zero) == zero:
                    instruction_pointer = program[instruction_pointer + 1].value
                else:
                    instruction_pointer += 2
            elif program[instruction_pointer] == F(']'):
                if memory.get(memory_pointer, zero) != zero:
                    instruction_pointer = program[instruction_pointer + 1].value
                else:
                    instruction_pointer += 2
            elif program[instruction_pointer] == F('<'):
                instruction_pointer += 1
                memory_pointer -= one
            elif program[instruction_pointer] == F('>'):
                instruction_pointer += 1
                memory_pointer += one
            elif program[instruction_pointer] == F('+'):
                instruction_pointer += 1
                memory[memory_pointer] = memory.get(memory_pointer, zero) + one
            elif program[instruction_pointer] == F('-'):
                instruction_pointer += 1
                memory[memory_pointer] = memory.get(memory_pointer, zero) - one
            elif program[instruction_pointer] == F('.'):
                instruction_pointer += 1
                output_data += chr(int(memory[memory_pointer].value % 256))
            elif program[instruction_pointer] == F(','):
                instruction_pointer += 1
                if input_data:
                    char = input_data[input_counter]
                    input_counter += 1
                else:
                    char = sys.stdin.read(1)
                memory[memory_pointer] = BaseFieldElement(ord(char), field)
            else:
                assert(
                    False), f"unrecognized instruction at {instruction_pointer}: {program[instruction_pointer].value}"

        return input_data, output_data

    def simulate(program, input_data=[]):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        two = BaseFieldElement(2, field)
        def F(x): return BaseFieldElement(ord(x), field)

        # initial state
        register = Register()
        register.current_instruction = program[0]
        memory = dict()  # field elements to field elements
        input_counter = 0
        output_data = []

        # prepare tables
        processor_table = ProcessorTable(field)
        processor_table.table = [[register.cycle, register.instruction_pointer, register.current_instruction,
                  register.next_instruction, register.memory_pointer, register.memory_value, register.is_zero]]

        memory_table = MemoryTable(field)
        memory_table.table = [
            [register.cycle, register.memory_pointer, register.memory_value]]

        instruction_table = InstructionTable(field)
        instruction_table.table = [[BaseFieldElement(i, field), program[i]] for i in range(
            len(program))] + [[register.instruction_pointer, register.current_instruction]]

        previous_input_value = zero
        input_table = InputTable(field)

        previous_output_value = zero
        output_table = OutputTable(field)

        # main loop
        while register.instruction_pointer.value < len(program):
            # record changes, to be used if necessary
            old_memory_value = register.memory_value

            # update pointer registers according to instruction
            if register.current_instruction == F('['):
                if register.memory_value == zero:
                    register.instruction_pointer = program[register.instruction_pointer.value + 1]
                else:
                    register.instruction_pointer += two

            elif register.current_instruction == F(']'):
                if register.memory_value != zero:
                    register.instruction_pointer = program[register.instruction_pointer.value + 1]
                else:
                    register.instruction_pointer += two

            elif register.current_instruction == F('<'):
                register.instruction_pointer += one
                register.memory_pointer -= one

            elif register.current_instruction == F('>'):
                register.instruction_pointer += one
                register.memory_pointer += one

            elif register.current_instruction == F('+'):
                register.instruction_pointer += one
                memory[register.memory_pointer] = memory.get(
                    register.memory_pointer, zero) + one

            elif register.current_instruction == F('-'):
                register.instruction_pointer += one
                memory[register.memory_pointer] = memory.get(
                    register.memory_pointer, zero) - one

            elif register.current_instruction == F('.'):
                register.instruction_pointer += one
                output_table.table += [[memory[register.memory_pointer],
                                  previous_output_value]]
                previous_output_value = memory[register.memory_pointer]
                output_data += chr(
                    int(memory[register.memory_pointer].value % 256))

            elif register.current_instruction == F(','):
                register.instruction_pointer += one
                char = input_data[input_counter]
                input_counter += 1
                memory[register.memory_pointer] = BaseFieldElement(
                    ord(char), field)
                input_table.table += [[memory[register.memory_pointer],
                                 previous_input_element]]
                previous_input_element = memory[register.memory_pointer]

            else:
                assert(
                    False), f"unrecognized instruction at {register.instruction_pointer.value}: '{chr(register.current_instruction.value)}'"

            # update non-pointer registers
            register.cycle += one
            if register.instruction_pointer.value < len(program):
                register.current_instruction = program[register.instruction_pointer.value]
            else:
                register.current_instruction = zero
            if register.instruction_pointer.value < len(program)-1:
                register.next_instruction = program[register.instruction_pointer.value + 1]
            else:
                register.next_instruction = zero
            register.memory_value = memory.get(register.memory_pointer, zero)

            # collect values to add new rows in execution tables
            processor_table.table += [[register.cycle, register.instruction_pointer, register.current_instruction,
                       register.next_instruction, register.memory_pointer, register.memory_value, register.is_zero]]
            memory_table.table += [[register.cycle,
                              register.memory_pointer, register.memory_value]]

            if register.instruction_pointer.value < len(program):
                instruction_table.table += [[register.instruction_pointer,
                                       register.current_instruction]]
            else:
                instruction_table.table += [[register.instruction_pointer, zero]]

        # post-process context tables
        # sort by memory address
        memory_table.table.sort(key=lambda row: row[1].value)
        # sort by instruction address
        instruction_table.table.sort(key=lambda row: row[0].value)

        return processor_table, instruction_table, memory_table, input_table, output_table

    @staticmethod
    def num_challenges():
        return 10

    @staticmethod
    def instruction_transition_constraints(instruction):
        # register names
        cycle = 0
        instruction_pointer = 1
        current_instruction = 2
        next_instruction = 3
        memory_pointer = 4
        memory_value = 5
        is_zero = 6
        nextt = 7

        # useful algebraic shorthands
        field = BaseField.main()
        zero = MPolynomial.constant(field.zero())
        one = MPolynomial.constant(field.one())
        two = MPolynomial.constant(BaseFieldElement(2, field))
        x = MPolynomial.variables(14, field)
        polynomials = []

        # set constraints
        polynomials = []

        if instruction == '[':
            # if memval is zero, jump; otherwise skip one
            polynomials += [x[memory_value] * (x[instruction_pointer + nextt] - x[instruction_pointer] - two) +
                            x[is_zero] * (x[instruction_pointer + nextt] - x[next_instruction])]
            # other registers update normally
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one
            # memory pointer does not change
            polynomials += [x[memory_pointer + nextt] - x[memory_pointer]]
            # memory value does not change
            polynomials += [x[memory_value + nextt] - x[memory_value]]
            # truth of "memval==0" does not change
            polynomials += [x[is_zero + nextt] - x[is_zero]]

        elif instruction == ']':
            # if memval is nonzero, jump; otherwise skip one
            polynomials += [x[is_zero] * (x[instruction_pointer + nextt] - x[instruction_pointer] - two) +
                            x[memory_value] * (x[instruction_pointer + nextt] - x[next_instruction])]
            # other registers update normally
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one
            # memory pointer does not change
            polynomials += [x[memory_pointer + nextt] - x[memory_pointer]]
            # memory value does not change
            polynomials += [x[memory_value + nextt] - x[memory_value]]
            # truth of "memval==0" does not change
            polynomials += [x[is_zero + nextt] - x[is_zero]]

        elif instruction == '<':
            # decrease memory pointer
            polynomials += [x[memory_pointer + nextt] -
                            x[memory_pointer] + one]
            # new memory value is unconstrained
            # but new memval==0 is not
            polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
            polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
            # other registers update normally
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # increase instruction pointer by one
            polynomials += [x[instruction_pointer + nextt] -
                            x[instruction_pointer] + one]

        elif instruction == '>':
            # decrease memory pointer
            polynomials += [x[memory_pointer + nextt] -
                            x[memory_pointer] - one]
            # new memory value is unconstrained
            # but new memval==0 is not
            polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
            polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
            # other registers update normally
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # increase instruction pointer by one
            polynomials += [x[instruction_pointer + nextt] -
                            x[instruction_pointer] + one]

        elif instruction == '+':
            # increase memory value
            polynomials += [x[memory_value + nextt] - x[memory_value] - one]
            # re-set memval==0
            polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
            polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
            # other registers update normally
            # memory pointer stays the same
            polynomials += [x[memory_pointer + nextt] - x[memory_pointer]]
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # increase instruction pointer by one
            polynomials += [x[instruction_pointer + nextt] -
                            x[instruction_pointer] + one]

        elif instruction == '-':
            # increase memory value
            polynomials += [x[memory_value + nextt] - x[memory_value] + one]
            # re-set memval==0
            polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
            polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
            # other registers update normally
            # memory pointer stays the same
            polynomials += [x[memory_pointer + nextt] - x[memory_pointer]]
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # increase instruction pointer by one
            polynomials += [x[instruction_pointer + nextt] -
                            x[instruction_pointer] + one]

        elif instruction == ',':
            # new memory value is unconstrained
            # re-set memval==0
            polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
            polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
            # other registers update normally
            # memory pointer stays the same
            polynomials += [x[memory_pointer + nextt] - x[memory_pointer]]
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # increase instruction pointer by one
            polynomials += [x[instruction_pointer + nextt] -
                            x[instruction_pointer] + one]

        elif instruction == '.':
            # all registers update normally
            polynomials += [x[memory_value + nextt] -
                            x[memory_value]]  # no change in memval
            # no change in memval==0
            polynomials += [x[is_zero + nextt] - x[is_zero]]
            # memory pointer stays the same
            polynomials += [x[memory_pointer + nextt] - x[memory_pointer]]
            # increase cycle count by one
            polynomials += [x[cycle + nextt] - x[cycle] + one]
            # increase instruction pointer by one
            polynomials += [x[instruction_pointer + nextt] -
                            x[instruction_pointer] + one]

        else:
            assert(False), "given instruction not in instruction set"

        return polynomials

    # returns a polynomial in X, that evaluates to 0 in all instructions except the argument
    @staticmethod
    def instruction_picker(X, instruction):
        acc = MPolynomial.constant(VirtualMachine.field.one())
        field = VirtualMachine.field
        for c in "[]<>+-.,":
            if c == instruction:
                pass
            acc = acc * \
                (X - MPolynomial.constant(BaseFieldElement(ord(c), field)))
        return acc

    def processor_transition_constraints(challenges=None):
        # register names
        cycle = 0
        instruction_pointer = 1
        current_instruction = 2
        next_instruction = 3
        memory_pointer = 4
        memory_value = 5
        is_zero = 6

        if challenges == None:
            nextt = 7
        else:
            instruction_permutation = 7
            memory_permutation = 8
            input_evaluation = 9
            input_indeterminate = 10
            output_evaluation = 11
            output_indeterminate = 12
            nextt = 13

        # build polynomials
        field = BaseField.main()
        x = MPolynomial.variables(2*nextt, field)

        airs = []
        for c in "[]<>+-,.":
            instruction_vector = VirtualMachine.instruction_transition_constraints(
                c)
            instruction_picker = VirtualMachine.instruction_picker(
                x[current_instruction], c)
            for instr in instruction_vector:
                airs += [instruction_picker * instr]

        # if challenges are supplied, include polynomials for extended table
        if challenges != None:
            challenges = [MPolynomial.constant(c) for c in challenges]

            # names for challenges
            a = challenges[0]
            b = challenges[1]
            c = challenges[2]
            d = challenges[3]
            e = challenges[4]
            f = challenges[5]
            alpha = challenges[6]
            beta = challenges[7]
            gamma = challenges[8]
            delta = challenges[9]

            # 1. running product for instruction permutation
            airs += [x[instruction_permutation] * (alpha - a * x[instruction_pointer + nextt] - b *
                                                   x[current_instruction + nextt] - c * x[next_instruction + nextt]) - x[instruction_permutation + nextt]]

            # 2. running product for memory access
            airs += [x[memory_permutation] * (beta - d * x[cycle + nextt] - e * x[memory_pointer +
                                              nextt] - f * x[memory_value + nextt]) - x[memory_permutation + nextt]]

            # 3. evaluation for input
            selector = x[current_instruction] - \
                MPolynomial.constant(BaseFieldElement(ord(','), field))
            inverse_selector = MPolynomial.constant(BaseFieldElement(1, field))
            for c in "[]<>+-.":
                inverse_selector *= x[current_instruction] - \
                    MPolynomial.constant(BaseFieldElement(ord(c), field))

            airs += [inverse_selector * (-x[input_evaluation + nextt] + x[input_evaluation] + x[input_indeterminate+nextt]
                                         * x[memory_value+nextt]) + selector * (x[input_evaluation + nextt] - x[input_evaluation])]
            airs += [inverse_selector * (x[input_indeterminate + nextt] - x[input_indeterminate]
                                         * gamma) + selector * (x[input_indeterminate + nextt] - x[input_indeterminate])]

            # 4. evaluation for output
            selector = x[current_instruction] - \
                MPolynomial.constant(BaseFieldElement(ord('.'), field))
            inverse_selector = MPolynomial.constant(BaseFieldElement(1, field))
            for c in "[]<>+-,":
                inverse_selector *= x[current_instruction] - \
                    MPolynomial.constant(BaseFieldElement(ord(c), field))

            airs += [inverse_selector * (-x[output_evaluation + nextt] + x[output_indeterminate + nextt] * x[memory_value +
                                         nextt] + x[output_evaluation]) + selector * (x[output_evaluation] - x[output_evaluation + nextt])]
            airs += [inverse_selector * (x[output_indeterminate + nextt] - delta * x[output_indeterminate]) +
                     selector * (x[output_indeterminate + nextt] - x[output_indeterminate])]

        return airs

    def processor_boundary_constraints(challenges=None):
        # format: (register, cycle, value)
        constraints = [(0, 0, VirtualMachine.field.zero()),  # cycle
                       # instruction pointer
                       (1, 0, VirtualMachine.field.zero()),
                       # (2, 0, ???), # current instruction
                       # (3, 0, ???), # next instruction
                       (4, 0, VirtualMachine.field.zero()),  # memory pointer
                       (5, 0, VirtualMachine.field.zero()),  # memory value
                       (6, 0, VirtualMachine.field.one())]  # memval==0

        if challenges != None:
            constraints += [(7, 0, VirtualMachine.field.one()),  # instruction permutation
                            # memory permutation
                            (8, 0, VirtualMachine.field.one()),
                            # input indeterminate
                            (9, 0, VirtualMachine.field.one()),
                            # input evaluation
                            (10, 0, VirtualMachine.field.zero()),
                            # output indeterminate
                            (11, 0, VirtualMachine.field.one()),
                            (12, 0, VirtualMachine.field.zero())]  # output evaluation
        return constraints

    def instruction_table_transition_constraints():
        # column names
        instruction_pointer = 0
        instruction_value = 1
        nextt = 2

        # build polynomials
        field = BaseField.main()
        x = MPolynomial.variables(4, field)
        one = MPolynomial.constant(field.one())
        airs = []

        # constraints:
        # 1. instruction pointer increases by at most one
        # (DNF:) <=>. IP* = IP \/ IP* = IP+1
        airs += [(x[instruction_pointer+nextt] - x[instruction_pointer])
                 * (x[instruction_pointer+nextt] - x[instruction_pointer] - one)]

        # 2. if the instruction pointer is the same, then the instruction value must be the same also
        #        <=> IP*=IP => IV*=IV
        # (DNF:) <=> IP*=/=IP \/ IV*=IV
        airs += [(x[instruction_pointer+nextt] - x[instruction_pointer] - one)
                 * (x[instruction_value+nextt] - x[instruction_value])]

        return airs

    def instruction_table_boundary_constraints():
        return [(0, 0, VirtualMachine.field.zero()),  # instruction pointer
                # (1, 0, ???), # matching instruction
                ]

    def memory_transition_constraints():
        # column names
        cycle = 0
        memory_pointer = 1
        memory_value = 2
        nextt = 3

        # build polynomials
        field = BaseField.main()
        x = MPolynomial.variables(6, field)
        one = MPolynomial.constant(field.one())
        airs = []

        # constraints

        # 1. memory pointer increases by one or zero
        # <=>. (MP*=MP+1) \/ (MP*=MP)
        airs += [(x[memory_pointer + nextt] - x[memory_pointer] - one)
                 * (x[memory_pointer + nextt] - x[memory_pointer])]

        # 2. if memory pointer increases by zero, then memory value can change only if cycle counter increases by one
        #        <=>. MP*=MP => (MV*=/=MV => CLK*=CLK+1)
        #        <=>. MP*=/=MP \/ (MV*=/=MV => CLK*=CLK+1)
        # (DNF:) <=>. MP*=/=MP \/ MV*=MV \/ CLK*=CLK+1
        airs += [(x[memory_pointer+nextt] - x[memory_pointer] - one) *
                 (x[memory_value + nextt] - x[memory_value]) * (x[cycle+nextt] - x[cycle] - one)]

        # 3. if memory pointer increases by one, then memory value must be set to zero
        #        <=>. MP*=MP+1 => MV* = 0
        # (DNF:) <=>. MP*=/=MP+1 \/ MV*=0
        airs += [(x[memory_pointer + nextt] - x[memory_pointer])
                 * x[memory_value + nextt]]

        return airs

    def memory_boundary_constraints():
        return [(0, 0, VirtualMachine.field.zero()),  # cycle
                (1, 0, VirtualMachine.field.zero()),  # memory pointer
                (2, 0, VirtualMachine.field.zero()),  # memory value
                ]

    def io_transition_constraints():
        # column names
        current_value = 0
        previous_value = 1
        nextt = 2

        # build polynomials
        field = BaseField.main()
        x = MPolynomial.variables(4, field)
        one = MPolynomial.constant(field.one())
        airs = []

        # constraints

        # 1. next previous_value is current current_value
        # <=>. PV* = CV
        return [x[previous_value + nextt] - x[current_value]]

    def io_boundary_constraints():
        return [(1, 0, VirtualMachine.field.zero())]  # column, row, value

    def processor_table_extend(processor_table, challenges):
        # register names
        cycle = 0
        instruction_pointer = 1
        current_instruction = 2
        next_instruction = 3
        memory_pointer = 4
        memory_value = 5
        is_zero = 6

        # names for challenges
        a = challenges[0]
        b = challenges[1]
        c = challenges[2]
        d = challenges[3]
        e = challenges[4]
        f = challenges[5]
        alpha = challenges[6]
        beta = challenges[7]
        gamma = challenges[8]
        delta = challenges[9]

        # algebra stuff
        field = VirtualMachine.field
        one = field.one()
        zero = field.zero()

        # prepare for loop
        instruction_permutation_running_product = one
        memory_permutation_running_product = one
        input_evaluation = zero
        input_indeterminate = one
        output_evaluation = zero
        output_indeterminate = one

        # loop over all rows
        table_extension = []
        for row in processor_table:
            new_row = []

            # next, define the additional columns

            # 1. running product for instruction permutation
            instruction_permutation_running_product *= alpha - a * \
                row[instruction_pointer] - b * \
                row[current_instruction] - c * row[next_instruction]
            new_row += [instruction_permutation_running_product]

            # 2. running product for memory access
            memory_permutation_running_product *= beta - d * \
                row[cycle] - e * row[memory_pointer] - f * row[memory_value]
            new_row += [memory_permutation_running_product]

            # 3. evaluation for input
            if row[current_instruction] == BaseFieldElement(ord(','), field):
                input_evaluation += input_indeterminate * row[memory_value]
                input_indeterminate *= gamma
            new_row += [input_indeterminate]
            new_row += [input_evaluation]

            # 4. evaluation for output
            if row[current_instruction] == BaseFieldElement(ord('.'), field):
                output_evaluation += output_indeterminate * row[memory_value]
                output_indeterminate *= delta
            new_row += [output_indeterminate]
            new_row += [output_evaluation]

            table_extension += [new_row]

        return table_extension

    def instruction_table_extend(instruction_table, challenges):

        # names for challenges
        a = challenges[0]
        b = challenges[1]
        c = challenges[2]
        d = challenges[3]
        e = challenges[4]
        f = challenges[5]
        alpha = challenges[6]
        beta = challenges[7]
        gamma = challenges[8]
        delta = challenges[9]

        # algebra stuff
        field = VirtualMachine.field
        one = field.one()

        # prepare loop
        extended_instruction_table = []
        instruction_permutation_running_product = one

        # loop over all rows of table
        for i in range(len(instruction_table)):
            row = instruction_table[i]
            new_row = []

            # match with this:
            # 1. running product for instruction permutation
            #instruction_permutation_running_product *= alpha - a * row[instruction_pointer] - b * row[current_instruction] - c * row[next_instruction]
            #new_row += [[instruction_permutation_running_product]]

            current_instruction = instruction_table[i][1]
            if i < len(instruction_table)-1:
                next_instruction = instruction_table[i+1][1]
            else:
                next_instruction = 0
            instruction_permutation_running_product *= alpha - a * \
                row[0] - b * current_instruction - c * next_instruction
            new_row += [[instruction_permutation_running_product]]

            extended_instruction_table += [new_row]

        return extended_instruction_table

    def memory_table_extend(memory_table, challenges):
        # names for challenges
        a = challenges[0]
        b = challenges[1]
        c = challenges[2]
        d = challenges[3]
        e = challenges[4]
        f = challenges[5]
        alpha = challenges[6]
        beta = challenges[7]
        gamma = challenges[8]
        delta = challenges[9]

        # algebra stuff
        field = VirtualMachine.field
        one = field.one()

        # prepare loop
        extended_memory_table = []
        memory_permutation_running_product = one

        # loop over all rows of table
        for row in memory_table:
            new_row = []

            # match with this:
            # 2. running product for memory access
            #memory_permutation_running_product *= beta - d * row[cycle] - e * row[memory_pointer] - f * row[memory_value]
            #new_row += [[memory_permutation_running_product]]
            memory_permutation_running_product *= beta - \
                d * row[0] - e * row[1] - f * row[2]

            new_row += [memory_permutation_running_product]

            extended_memory_table += [new_row]

        return extended_memory_table

    def input_table_extend(input_table, challenges):
        # names for challenges
        a = challenges[0]
        b = challenges[1]
        c = challenges[2]
        d = challenges[3]
        e = challenges[4]
        f = challenges[5]
        alpha = challenges[6]
        beta = challenges[7]
        gamma = challenges[8]
        delta = challenges[9]

        # algebra stuff
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()

        # prepare loop
        extended_input_table = []
        input_running_evaluation = zero
        input_running_indeterminate = one

        # loop over all rows of table
        for row in input_table:
            new_row = []

            # match with this:
            # 3. evaluation for input
            # if row[current_instruction] == BaseFieldElement(ord(','), field):
            #    input_evaluation += input_indeterminate * row[memory_value]
            #    input_indeterminate *= gamma
            #new_row += [input_indeterminate]
            #new_row += [input_evaluation]

            input_running_evaluation += input_running_indeterminate * row[0]
            input_running_indeterminate *= gamma

            new_row += [input_running_indeterminate]
            new_row += [input_running_evaluation]

            extended_input_table += [new_row]

        return extended_input_table

    def output_table_extend(output_table, challenges):
        # names for challenges
        a = challenges[0]
        b = challenges[1]
        c = challenges[2]
        d = challenges[3]
        e = challenges[4]
        f = challenges[5]
        alpha = challenges[6]
        beta = challenges[7]
        gamma = challenges[8]
        delta = challenges[9]

        # algebra stuff
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()

        # prepare loop
        extended_output_table = []
        output_running_evaluation = zero
        output_running_indeterminate = one

        # loop over all rows of table
        for row in output_table:
            new_row = []

            # match with this:
            # 4. evaluation for output
            # if row[current_instruction] == BaseFieldElement(ord('.'), field):
            #    output_evaluation += output_indeterminate * row[memory_value]
            #    output_indeterminate *= delta
            #new_row += [output_indeterminate]
            #new_row += [output_evaluation]

            output_running_evaluation += output_running_indeterminate * row[0]
            output_running_indeterminate *= gamma

            new_row += [output_running_indeterminate]
            new_row += [output_running_evaluation]

            extended_output_table += [new_row]

        return extended_output_table
