from algebra import *
from io_table import IOTable
from instruction_table import InstructionTable
from memory_table import MemoryTable
from multivariate import *
import sys

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
                program[stack[-1]] = BaseFieldElement(len(program), field)
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

    @staticmethod
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
        # Programs shorter than two instructions aren't valid programs.
        register.next_instruction = program[1]
        memory = dict()  # field elements to field elements
        input_counter = 0
        output_data = []

        # prepare tables
        processor_table = ProcessorTable(field)
        processor_table.table = []

        memory_table = MemoryTable(field)
        memory_table.table = []

        instruction_table = InstructionTable(field)
        instruction_table.table = [[BaseFieldElement(i, field), program[i], program[i+1]] for i in range(len(program)-1)] + \
                                  [[BaseFieldElement(
                                      len(program)-1, field), program[-1], field.zero()]]

        input_table = IOTable(field)

        output_table = IOTable(field)

        # main loop
        while register.instruction_pointer.value < len(program):
            # collect values to add new rows in execution tables
            processor_table.table += [[register.cycle,
                                       register.instruction_pointer,
                                       register.current_instruction,
                                       register.next_instruction,
                                       register.memory_pointer,
                                       register.memory_value,
                                       register.is_zero]]

            memory_table.table += [[register.cycle,
                                    register.memory_pointer,
                                    register.memory_value]]

            instruction_table.table += [[register.instruction_pointer,
                                         register.current_instruction,
                                         register.next_instruction]]

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
                output_table.table += [[memory.get(register.memory_pointer, zero)]]
                output_data += chr(
                    int(memory.get(register.memory_pointer,zero).value % 256))

            elif register.current_instruction == F(','):
                register.instruction_pointer += one
                char = input_data[input_counter]
                input_counter += 1
                memory[register.memory_pointer] = BaseFieldElement(
                    ord(char), field)
                input_table.table += [[memory[register.memory_pointer]]]

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
            if register.memory_value.is_zero():
                register.is_zero = one
            else:
                register.is_zero = zero
        
        # collect final state into execution tables
        processor_table.table += [[register.cycle,
                                    register.instruction_pointer,
                                    register.current_instruction,
                                    register.next_instruction,
                                    register.memory_pointer,
                                    register.memory_value,
                                    register.is_zero]]

        memory_table.table += [[register.cycle,
                                register.memory_pointer,
                                register.memory_value]]

        instruction_table.table += [[register.instruction_pointer,
                                        register.current_instruction,
                                        register.next_instruction]]

        # post-process context tables
        # sort by memory address
        memory_table.table.sort(key=lambda row: row[1].value)
        # sort by instruction address
        instruction_table.table.sort(key=lambda row: row[0].value)

        # compute instance data for computation
        log_time = 0
        while 1 << log_time < len(processor_table.table):
            log_time += 1

        return log_time, processor_table, instruction_table, memory_table, input_table, output_table

    @staticmethod
    def num_challenges():
        return 11

    @staticmethod
    def evaluation_terminal(table, alpha):
        xfield = alpha.field
        acc = xfield.zero()
        for i in range(len(table)):
            t = table[i]
            acc = alpha * acc + xfield.lift(t[0])
        return acc

    @staticmethod
    def program_evaluation(program, a, b, c, eta):
        field = program[0].field
        xfield = a.field
        running_sum = xfield.zero()
        previous_address = -xfield.one()
        padded_program = [xfield.lift(p)
                          for p in program] + [xfield.zero()]
        for i in range(len(padded_program)-1):
            address = xfield.lift(BaseFieldElement(i, field))
            current_instruction = padded_program[i]
            next_instruction = padded_program[i+1]
            if previous_address != address:
                running_sum = running_sum * eta + a * address + \
                    b * current_instruction + c * next_instruction
            previous_address = address

        index = len(padded_program)-1
        address = xfield.lift(BaseFieldElement(index, field))
        current_instruction = padded_program[index]
        next_instruction = xfield.zero()
        running_sum = running_sum * eta + a * address + b * current_instruction + c * next_instruction

        return running_sum

    @staticmethod
    def program_permutation_cofactor(program, a, b, c, alpha, initial_value=None):
        field = program[0].field
        xfield = a.field
        if not initial_value:
            running_product = xfield.one()
        else:
            running_product = initial_value
        padded_program = [xfield.lift(p)
                          for p in program] + [xfield.zero()]
        for i in range(len(padded_program)-1):
            address = xfield.lift(BaseFieldElement(i, field))
            current_instruction = padded_program[i]
            next_instruction = padded_program[i+1]
            running_product *= alpha - a * address - b * \
                current_instruction - c * next_instruction

        return running_product
