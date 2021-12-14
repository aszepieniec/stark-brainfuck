from algebra import *
import sys

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
                else
                    char = sys.stdin.read(1)
                memory[memory_pointer] = BaseFieldElement(ord(char), field)
            else:
                assert(False), f"unrecognized instruction at {instruction_pointer}: {program[instruction_pointer].value}"

        return input_data, output_data

    def simulate( program, input_data ):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        two = BaseFieldElement(2, field)
        F = lambda x : BaseFieldElement(ord(x), field)

        # register names
        cycle = 0
        instruction = 1
        instruction_pointer = 2
        memory_value = 3

        # initial state
        instruction_pointer = 0
        memory_pointer = zero
        register = [zero] * 4
        register[instruction] = program[0]
        memory = dict() # field elements to field elements
        input_counter = 0

        # main loop
        trace = [register]
        while register[instruction_pointer].value < len(program):
            # update register according to instruction
            if register[instruction] == F('['):
                if register[memory_value] == zero:
                    register[instruction_pointer] = program[instruction_pointer.value + 1]
                else:
                    register[instruction_pointer] += two
                register[cycle] += one
                register[instruction] = program[instruction_pointer.value]
            elif register[instruction] == F(']'):
                if register[memory_value] != zero:
                    register[instruction_pointer] = program[instruction_pointer.value + 1]
                else:
                    register[instruction_pointer] += two
                register[cycle] += one
                register[instruction] = program[instruction_pointer.value]
            elif register[instruction] == F('<'):
                register[instruction_pointer] += one
                register[memory_pointer] -= one
                register[cycle] += one
                register[memory_value] = memory.get(register[memory_pointer], zero)
                register[instruction] = program[instruction_pointer.value]
            elif register[instruction] == F('>'):
                register[instruction_pointer] += one
                register[memory_pointer] += one
                register[cycle] += one
                register[memory_value] = memory.get(register[memory_pointer], zero)
                register[instruction] = program[instruction_pointer.value]
            elif register[instruction] == F('+'):
                register[instruction_pointer] += one
                memory[memory_pointer] = memory.get(memory_pointer, zero) + one
                register[cycle] += one
                register[memory_value] += one
                register[instruction] = program[instruction_pointer.value]
            elif register[instruction] == F('-'):
                register[instruction_pointer] += one
                memory[memory_pointer] = memory.get(memory_pointer, zero) - one
                register[cycle] += one
                register[memory_value] -= one
                register[instruction] = program[instruction_pointer.value]
            elif register[instruction] == F('.'):
                register[instruction_pointer] += one
                output_data += chr(int(memory[memory_pointer].value % 256))
                register[cycle] += one
                register[instruction] = program[instruction_pointer.value]
            elif register[instruction] == F(','):
                register[instruction_pointer] += one
                char = input_data[input_counter]
                input_counter += 1
                memory[memory_pointer] = BaseFieldElement(ord(char), field)
                register[memory_value] = memory[memory_pointer]
                register[cycle] += one
                register[instruction] = program[instruction_pointer.value]
            else:
                assert(False), f"unrecognized instruction at {instruction_pointer}: {program[instruction_pointer].value}"

            # collect values
            trace += [register]

        return input_data, output_data

