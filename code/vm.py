from algebra import *
from multivariate import *
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

        # prepare tables
        trace = [[register.cycle, register.instruction_pointer, register.current_instruction, register.next_instruction, register.memory_pointer, register.memory_value, register.is_zero]]
        memory_table = [[register.cycle, register.memory_pointer, register.memory_value]]
        instruction_table = [[BaseFieldElement(i, field), program[i]] for i in range(len(program))] + [[register.instruction_pointer, register.current_instruction]]
        previous_input_value = zero
        input_table = []
        previous_output_value = zero
        output_table = []

        # main loop
        while register.instruction_pointer.value < len(program):
            # record changes, to be used if necessary
            old_memory_value = register.memory_value

            # updates common to all instructions
            register.current_instruction = program[register.instruction_pointer.value]
            if register.instruction_pointer.value < len(program)-1:
                register.next_instruction = program[register.instruction_pointer.value+1]
            else:
                register.next_instruction = zero
            register.cycle += one
            register.memory_value = memory.get(register.memory_pointer, zero)
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
                register.memory_pointer -= one

            elif register.current_instruction == F('>'):
                register.instruction_pointer += one
                register.memory_pointer += one

            elif register.current_instruction == F('+'):
                register.instruction_pointer += one
                memory[register.memory_pointer] = memory.get(register.memory_pointer, zero) + one

            elif register.current_instruction == F('-'):
                register.instruction_pointer += one
                memory[register.memory_pointer] = memory.get(register.memory_pointer, zero) - one

            elif register.current_instruction == F('.'):
                register.instruction_pointer += one
                output_table += [[memory[register.memory_pointer], previous_output_value]]
                previous_output_value = memory[register.memory_pointer]
                output_data += chr(int(memory[register.memory_pointer].value % 256))

            elif register.current_instruction == F(','):
                register.instruction_pointer += one
                char = input_data[input_counter]
                input_counter += 1
                memory[register.memory_pointer] = BaseFieldElement(ord(char), field)
                input_table += [[memory[register.memory_pointer], previous_input_element]]
                previous_input_element = memory[register.memory_pointer]

            else:
                assert(False), f"unrecognized instruction at {register.instruction_pointer.value}: '{chr(register.current_instruction.value)}'"

            # collect values to add new rows in execution tables
            trace += [[register.cycle, register.instruction_pointer, register.current_instruction, register.next_instruction, register.memory_pointer, register.memory_value, register.is_zero]]
            memory_table += [[register.cycle, register.memory_pointer, register.memory_value]]

            if register.instruction_pointer.value < len(program):
                instruction_table += [[register.instruction_pointer, program[register.instruction_pointer.value]]]
            else:
                instruction_table += [[register.instruction_pointer, zero]]

        # post-process context tables
        memory_table.sort(key = lambda row : row[1].value) # sort by memory address
        instruction_table.sort(key = lambda row : row[0].value) # sort by instruction address

        return output_data, trace, instruction_table, memory_table, input_table, output_table

    @staticmethod
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
        zero = MPolynomial.constant(field.zero())
        one = MPolynomial.constant(field.one())
        two = MPolynomial.constant(BaseFieldElement(2, field))
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
            polynomials += [x[memory_value + nextt] - x[memory_value] - one]
            # re-set memval==0
            polynomials += [x[is_zero + nextt] * (one - x[is_zero + nextt])]
            polynomials += [x[is_zero + nextt] * x[memory_value + nextt]]
            # other registers update normally
            polynomials += [x[memory_pointer + nextt] - x[memory_pointer]] # memory pointer stays the same
            polynomials += [x[cycle + nextt] - x[cycle] + one] # increase cycle count by one
            polynomials += [x[instruction_pointer + nextt] - x[instruction_pointer] + one] # increase instruction pointer by one
 
        elif instruction == '-':
            # increase memory value
            polynomials += [x[memory_value + nextt] - x[memory_value] + one]
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
    @staticmethod
    def instruction_picker( X, instruction ):
        acc = MPolynomial.constant(VirtualMachine.field.one())
        field = VirtualMachine.field
        for c in "[]<>+-.,":
            if c == instruction:
                pass
            acc = acc * (X - MPolynomial.constant(BaseFieldElement(ord(c), field)))
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

        airs = []
        for c in "[]<>+-,.":
            instruction_vector = VirtualMachine.instruction_transition_constraints(c)
            instruction_picker = VirtualMachine.instruction_picker(x[current_instruction], c)
            for instr in instruction_vector:
                airs += [instruction_picker * instr]

        return airs

    def processor_boundary_constraints( ):
        # format: (register, cycle, value)
        return [(0, 0, VirtualMachine.field.zero()), # cycle
            (1, 0, VirtualMachine.field.zero()), # instruction pointer
            #(2, 0, ???), # current instruction
            #(3, 0, ???), # next instruction
            (4, 0, VirtualMachine.field.zero()), # memory pointer
            (5, 0, VirtualMachine.field.zero()), # memory value
            (6, 0, VirtualMachine.field.one())] # memval==0

    def instruction_table_transition_constraints( ):
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
        airs += [(x[instruction_pointer+nextt] - x[instruction_pointer]) * (x[instruction_pointer+nextt] - x[instruction_pointer] - one)]

        # 2. if the instruction pointer is the same, then the instruction value must be the same also
        #        <=> IP*=IP => IV*=IV
        # (DNF:) <=> IP*=/=IP \/ IV*=IV
        airs += [(x[instruction_pointer+nextt] - x[instruction_pointer] - one) * (x[instruction_value+nextt] - x[instruction_value])]

        return airs

    def instruction_table_boundary_constraints( ):
        return [(0, 0, VirtualMachine.field.zero()), # instruction pointer
               #(1, 0, ???), # matching instruction
               ]

    def memory_transition_constraints( ):
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
        airs += [(x[memory_pointer + nextt] - x[memory_pointer] - one) * (x[memory_pointer + nextt] - x[memory_pointer])]

        # 2. if memory pointer increases by zero, then memory value can change only if cycle counter increases by one
        #        <=>. MP*=MP => (MV*=/=MV => CLK*=CLK+1)
        #        <=>. MP*=/=MP \/ (MV*=/=MV => CLK*=CLK+1)
        # (DNF:) <=>. MP*=/=MP \/ MV*=MV \/ CLK*=CLK+1
        airs += [(x[memory_pointer+nextt] - x[memory_pointer] - one) * (x[memory_value + nextt] - x[memory_value]) * (x[cycle+nextt] - x[cycle] - one)]

        # 3. if memory pointer increases by one, then memory value must be set to zero
        #        <=>. MP*=MP+1 => MV* = 0
        # (DNF:) <=>. MP*=/=MP+1 \/ MV*=0
        airs += [(x[memory_pointer + nextt] - x[memory_pointer]) * x[memory_value + nextt]]

        return airs

    def memory_boundary_constraints( ):
        return [(0, 0, VirtualMachine.field.zero()), # cycle
               (1, 0, VirtualMachine.field.zero()), # memory pointer
               (2, 0, VirtualMachine.field.zero()), # memory value
               ]

    def io_transition_constraints( ):
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

    def io_boundary_constraints( ):
        return [(1, 0, VirtualMachine.field.zero())]

