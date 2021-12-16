from algebra import *
import sys

class Register:
    field = BaseField.main()

    def __init__( self ):
        self.cycle = Register.field.zero()
        self.instruction_pointer = Register.field.zero()
        self.current_instruction = Register.field.zero()
        self.next_instruction = Register.field.zero()
        self.memory_pointer = Register.field.zero()
        self.memory_value = Register.field.zero()
        self.is_zero = Register.field.one()

class VirtualMachine:
    field = BaseField.main()

    def execute( brainfuck_code ):
        program = VirtualMachine.compile(brainfuck_code)
        input_data, output_data = VirtualMachine.perform(program)
        return input_data, output_data

    def compile( brainfuck_code ):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        F = lambda x : BaseFieldElement(ord(x), field)

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

    def perform( program, input_data=None ):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        F = lambda x : BaseFieldElement(ord(x), field)

        # initial state
        instruction_pointer = 0
        memory_pointer = BaseFieldElement(0, VirtualMachine.field)
        memory = dict() # field elements to field elements
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
                assert(False), f"unrecognized instruction at {instruction_pointer}: {program[instruction_pointer].value}"

        return input_data, output_data

    def simulate( program, input_data=[] ):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        two = BaseFieldElement(2, field)
        F = lambda x : BaseFieldElement(ord(x), field)

        # initial state
        register = Register()
        register.current_instruction = program[0]
        memory = dict() # field elements to field elements
        input_counter = 0
        output_data = []

        # main loop
        trace = [[register.cycle, register.instruction_pointer, register.current_instruction, register.next_instruction, register.memory_pointer, register.memory_value, register.is_zero]]
        while register.instruction_pointer.value < len(program):
            # common to all instructions
            register.current_instruction = program[register.instruction_pointer.value]
            if register.instruction_pointer.value < len(program)-1:
                register.next_instruction = program[register.instruction_pointer.value+1]
            else
                register.next_instruction = zero
            register.cycle += one
            register.memory_value = memory.get(register[memory_pointer], zero)
            if register.memory_value == zero:
                register.is_zero = one
            else:
                register.is_zero = zero

            # update register according to instruction
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
                register[memory_pointer] -= one

            elif register.current_instruction == F('>'):
                register.instruction_pointer += one
                register[memory_pointer] += one

            elif register.current_instruction == F('+'):
                register.instruction_pointer += one
                memory[register[memory_pointer]] = memory.get(register[memory_pointer], zero) + one

            elif register.current_instruction == F('-'):
                register.instruction_pointer += one
                memory[register[memory_pointer]] = memory.get(register[memory_pointer], zero) - one

            elif register.current_instruction == F('.'):
                register.instruction_pointer += one
                output_data += chr(int(memory[register[memory_pointer]].value % 256))

            elif register.current_instruction == F(','):
                register.instruction_pointer += one
                char = input_data[input_counter]
                input_counter += 1
                memory[register[memory_pointer]] = BaseFieldElement(ord(char), field)

            else:
                assert(False), f"unrecognized instruction at {register.instruction_pointer.value}: '{chr(register.current_instruction.value)}'"

            # collect values
            trace += [[register.cycle, register.instruction_pointer, register.current_instruction, register.next_instruction, register.memory_pointer, register.memory_value, register.is_zero]]

        return trace, output_data

        def instruction_transition_constraints( instruction ):
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
            zero = field.zero()
            one = field.one()
            two = BaseFieldElement(2, field)
            x = MPolynomial.variables(14, field)
            polynomials = []

            # set constraints
            polynomials = []

            if instruction == '[':
                # if memval is zero, jump; otherwise skip one
                polynomials += [x[memory_value] * (x[instruction_pointer + nextt] - x[instruction_pointer] - two) + x[is_zero] * (x[instruction_pointer + nextt] - x[next_instruction])]
                # other registers update normally
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                #polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer]] # memory pointer does not change
                polynomials += [x[memory_value + nextt] - x[memory_value]] # memory value does not change
                polynomials += [x[is_zero + nextt] - x[is_zero]] # truth of "memval==0" does not change

            elif instruction == ']':
                # if memval is nonzero, jump; otherwise skip one
                polynomials += [x[is_zero] * (x[instruction_pointer + nextt] - x[instruction_pointer] - two) + x[memory_value] * (x[instruction_pointer + nextt] - x[next_instruction])]
                # other registers update normally
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                #polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer]] # memory pointer does not change
                polynomials += [x[memory_value + nextt] - x[memory_value]] # memory value does not change
                polynomials += [x[is_zero + nextt] - x[is_zero]] # truth of "memval==0" does not change

            elif instruction == '<':
                # decrease memory pointer
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer] + one]
                # new memory value is unconstrained
                # but new memval==0 is not
                polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
                polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
                # other registers update normally
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one

            elif instruction == '>':
                # decrease memory pointer
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer] - one]
                # new memory value is unconstrained
                # but new memval==0 is not
                polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
                polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
                # other registers update normally
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one

            elif instruction == '+':
                # increase memory value
                polynomials += [x[memory_value + nextt] - x[memory_value] - 1]
                # re-set memval==0
                polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
                polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
                # other registers update normally
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer]] # memory pointer stays the same
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one

            elif instruction == '-':
                # increase memory value
                polynomials += [x[memory_value + nextt] - x[memory_value] + 1]
                # re-set memval==0
                polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
                polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
                # other registers update normally
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer]] # memory pointer stays the same
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one

            elif instruction == ',':
                # new memory value is unconstrained
                # re-set memval==0
                polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
                polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
                # other registers update normally
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer]] # memory pointer stays the same
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one

            elif instruction == '.':
                # all registers update normally
                polynomials += [x[memory_value + nextt] - x[memory_value]] # no change in memval
                polynomials += [x[is_zero + nextt] - x[is_zero]] # no change in memval==0
                polynomials += [x[memory_pointer + nextt] - x[memory_pointer]] # memory pointer stays the same
                polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
                polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one

            else:
                assert(False), "given instruction not in instruction set"

        return polynomials
                

        # returns a polynomial in X, that evaluates to 0 in all instructions except the argument
        def instruction_picker( X, instruction ):
            acc = X.ring_one()
            field = X.coefficients().values()[0].field
            for c in "[]<>+-.,":
                if c == instruction:
                    pass
                acc = acc * (X - MPolynomial({[0]: BaseFieldElement(ord(c), field)}))
            return acc

        def processor_transition_constraints( ):
            # register names
            cycle = 0
            instruction_pointer = 1
            current_instruction = 2
            next_instruction = 3
            memory_pointer = 4
            memory_value = 5
            is_zero = 6
            nextt = 7

            # build polynomials
            field = BaseField.main()
            x = MPolynomial.variables(14, field)
            polynomials = []

            airs = []
            for c in "[]<>+-,.":
                instruction_vector = VirtualMachine.instruction_transition_constraints(c)
                instruction_picker = VirtualMachine.instruction_picker(x[current_instruction], c)
                for instr in instruction_vector:
                    airs += [instruction_picker * instr]

            return airs

