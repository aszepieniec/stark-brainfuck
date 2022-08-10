from concurrent.futures import process
from brainfuck_stark import *
from os.path import exists
from vm import Register, VirtualMachine, getch


def mallorys_simulator(program, input_data=[]):
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
    if len(program) == 1:
        register.next_instruction = zero
    else:
        register.next_instruction = program[1]
    memory = dict()  # field elements to field elements
    input_counter = 0
    output_data = []

    # prepare tables
    processor_matrix = []
    instruction_matrix = [[BaseFieldElement(i, field), program[i], program[i+1]] for i in range(len(program)-1)] + \
        [[BaseFieldElement(
            len(program)-1, field), program[-1], field.zero()]]

    input_matrix = []
    output_matrix = []

    # main loop
    while register.instruction_pointer.value < len(program):
        # collect values to add new rows in execution tables
        processor_matrix += [[register.cycle,
                              register.instruction_pointer,
                              register.current_instruction,
                              register.next_instruction,
                              register.memory_pointer,
                              register.memory_value,
                              register.memory_value_inverse]]

        instruction_matrix += [[register.instruction_pointer,
                                register.current_instruction,
                                register.next_instruction]]

        # update pointer registers according to instruction
        if register.current_instruction == F('['):
            # This is the 1st part of the attack, a loop is *always* entered
            register.instruction_pointer += two

            # Original version is commented out below
            # if register.memory_value == zero:
            #     register.instruction_pointer = program[register.instruction_pointer.value + 1]
            # else:
            #     register.instruction_pointer += two
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
            output_matrix += [
                [memory.get(register.memory_pointer, zero)]]
            output_data += chr(
                int(memory.get(register.memory_pointer, zero).value % 256))

        elif register.current_instruction == F(','):
            register.instruction_pointer += one
            if input_data:
                char = input_data[input_counter]
                input_counter += 1
            else:
                char = getch()
            memory[register.memory_pointer] = BaseFieldElement(
                ord(char), field)
            input_matrix += [[memory[register.memory_pointer]]]

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
            register.memory_value_inverse = zero
        else:
            register.memory_value_inverse = register.memory_value.inverse()

        # This is the 2nd part of the attack
        if register.current_instruction == F('['):
            register.memory_value_inverse = BaseFieldElement(42, field)

    # collect final state into execution tables
    processor_matrix += [[register.cycle,
                          register.instruction_pointer,
                          register.current_instruction,
                          register.next_instruction,
                          register.memory_pointer,
                          register.memory_value,
                          register.memory_value_inverse]]

    instruction_matrix += [[register.instruction_pointer,
                            register.current_instruction,
                            register.next_instruction]]

    # post-process context tables
    # sort by instruction address
    instruction_matrix.sort(key=lambda row: row[0].value)

    # compute instance data for computation
    log_time = 0
    while 1 << log_time < len(processor_matrix):
        log_time += 1

    # order = 1 << 32
    # generator = field.primitive_nth_root(1<<32)
    # processor_table = ProcessorTable(field, len(processor_matrix), generator, order)
    # processor_table.table = processor_matrix
    # instruction_table = InstructionTable(field, len(instruction_matrix), generator, order)
    # memory_table = MemoryTable(field, len(memory_matrix), generator, order)
    # input_table = IOTable(field, len(input_matrix), generator, order)
    # output_table = IOTable(field, len(
    #     output_matrix), generator, order)

    return processor_matrix, instruction_matrix, input_matrix, output_matrix


def test_bfs():
    generator = BaseField.main().generator()
    xfield = ExtensionField.main()
    #program = VirtualMachine.compile(">>[++-]<++++++++")
    program = VirtualMachine.compile(
        ">++++++++++[>+++><<-]>+++><<>.................")
    # program = VirtualMachine.compile(",+.")
    # program = VirtualMachine.compile("++++++++++++++++++++.")
    # program = VirtualMachine.compile(",.........")
    # program = VirtualMachine.compile(",...")
    program = VirtualMachine.compile("++++")

    running_time, input_symbols, output_symbols = VirtualMachine.run(program)

    print("running time:", running_time)
    print("input symbols:", input_symbols)
    print("output_symbols:", output_symbols)

    # Print "Hello World!"
    # program = VirtualMachine.compile(
    #     "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.")
    processor_matrix, memory_matrix, instruction_matrix, input_matrix, output_matrix = VirtualMachine.simulate(
        program, input_data=input_symbols)
    assert (running_time == len(processor_matrix))
    memory_length = len(memory_matrix)

    bfs = BrainfuckStark(running_time, memory_length,
                         program, input_symbols, output_symbols)

    filename = "proof.dump"
    if exists(filename):
        fh = open(filename, "rb")
        proof = pickle.load(fh)
        fh.close()
    else:
        proof = bfs.prove(program, processor_matrix, memory_matrix,
                          instruction_matrix, input_matrix, output_matrix)
        fh = open(filename, "wb")
        pickle.dump(proof, fh)
        fh.close()

    # proof = bfs.prove(running_time, program, processor_matrix, instruction_matrix,
    #                   memory_matrix, input_matrix, output_matrix)

    # collapse matrix into list for input and output
    #input_symbols = [row[0] for row in input_matrix]
    #output_symbols = [row[0] for row in output_matrix]

    verdict = bfs.verify(proof)
    assert(verdict == True), "honest proof fails to verify"
    print("output length was:", len(output_symbols))

    if verdict == True:
        print("proof verified with output: \"" +
              "".join(output_symbols) + "\"")
    else:
        print("proof fails to verify with output:\"" +
              "".join(output_symbols) + "\"")


def set_adversarial_is_zero_value_test():
    program = VirtualMachine.compile("+>[++<-]")
    regular_processor_matrix, regular_instruction_matrix, regular_input_matrix, regular_output_matrix = VirtualMachine.simulate(
        program, input_data=[])
    regular_bfs = BrainfuckStark(
        len(regular_processor_matrix), program, [], [])
    regular_proof = regular_bfs.prove(len(regular_processor_matrix), program, regular_processor_matrix,
                                      regular_instruction_matrix, regular_input_matrix, regular_output_matrix)
    assert regular_bfs.verify(
        regular_proof), "Regular simulation must pass verifier"

    mallorys_processor_matrix, mallorys_instruction_matrix, mallorys_input_matrix, mallorys_output_matrix = mallorys_simulator(
        program, input_data=[])

    # Verify that the program is executed differently by the two simulators
    assert len(mallorys_processor_matrix) != len(
        regular_processor_matrix), "The execution trace of the regular and Mallory simulator must differ"
    assert regular_processor_matrix[-1][4] != mallorys_processor_matrix[-1][
        4], "Memory pointer must differ between regular simulation and Mallory's simulation"

    mallorys_bfs = BrainfuckStark(
        len(mallorys_processor_matrix), program, [], [])
    mallorys_proof = mallorys_bfs.prove(len(mallorys_processor_matrix), program, mallorys_processor_matrix,
                                        mallorys_instruction_matrix, mallorys_input_matrix, mallorys_output_matrix)
    success_of_mallory = mallorys_bfs.verify(mallorys_proof)
    assert not success_of_mallory, "Mallory's proof must fail to verify"
    print("Mallory's attack was defeated.")
    print("https://youtu.be/4aof9KxIJZo")
