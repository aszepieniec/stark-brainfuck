from algebra import *
import sys

class VirtualMachine:
    field = BaseField.main()

    def execute( brainfuck_code ):
        instructions_program = VirtualMachine.compile(brainfuck_code)
        input_data, output_data = VirtualMachine.perform(instructions_program)
        return input_data, output_data

    def compile( brainfuck_code ):
        # shorthands
        field = VirtualMachine.field
        zero = field.zero()
        one = field.one()
        F = lambda x : BaseFieldElement(ord(x), field)

        # parser
        instructions_program = []
        stack = []
        for symbol in brainfuck_code:
            instructions_program += [F(symbol)]
            if symbol == '[':
                instructions_program += [zero]
                stack += [len(instructions_program)-1]
            elif symbol == ']':
                instructions_program += [BaseFieldElement(stack[-1]+1, field)]
                instructions_program[stack[-1]] = BaseFieldElement(len(instructions_program)+1, field)
                stack = stack[:-1]
        return instructions_program

    def perform( instructions_program ):
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
        input_data = []

        # main loop
        while instruction_pointer < len(instructions_program):
            if instructions_program[instruction_pointer] == F('['):
                if memory.get(memory_pointer, zero) == zero:
                    instruction_pointer = instructions_program[instruction_pointer + 1].value
                else:
                    instruction_pointer += 2
            elif instructions_program[instruction_pointer] == F(']'):
                if memory.get(memory_pointer, zero) != zero:
                    instruction_pointer = instructions_program[instruction_pointer + 1].value
                else:
                    instruction_pointer += 2
            elif instructions_program[instruction_pointer] == F('<'):
                instruction_pointer += 1
                memory_pointer -= one
            elif instructions_program[instruction_pointer] == F('>'):
                instruction_pointer += 1
                memory_pointer += one
            elif instructions_program[instruction_pointer] == F('+'):
                instruction_pointer += 1
                memory[memory_pointer] = memory.get(memory_pointer, zero) + one
            elif instructions_program[instruction_pointer] == F('-'):
                instruction_pointer += 1
                memory[memory_pointer] = memory.get(memory_pointer, zero) - one
            elif instructions_program[instruction_pointer] == F('.'):
                instruction_pointer += 1
                output_data += chr(int(memory[memory_pointer].value % 256))
            elif instructions_program[instruction_pointer] == F(','):
                instruction_pointer += 1
                input_data += [sys.stdin.read(1)]
                memory[memory_pointer] = BaseFieldElement(input_data[-1], field)
            else:
                assert(False), f"unrecognized instruction at {instruction_pointer}: {instructions_program[instruction_pointer].value}"

        return input_data, output_data

      
