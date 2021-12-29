from extension import ExtensionField
from vm import *
import os

def test_vm( ):
    code = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
    expected_output = "Hello World!\n"
    #program = ">++++++++++[>+++><<-]>+++><<>."
    #expected_output = "!"
    input_data, output_data = VirtualMachine.execute(code)
    output_data = "".join(od for od in output_data)
    assert(output_data == expected_output), f"output data invalid; given:\"{output_data}\", but should be \"{expected_output}\""
    print(output_data)

def test_simulate( ):
    code = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
    expected_output = "Hello World!\n"
    #program = ">++++++++++[>+++><<-]>+++><<>."
    #expected_output = "!"
    program = VirtualMachine.compile(code)
    output_data, trace, instruction_table, memory_table, input_table, output_table = VirtualMachine.simulate(program)
    output_data = "".join(od for od in output_data)
    assert(output_data == expected_output), f"output data invalid; given:\"{output_data}\", but should be \"{expected_output}\""
    print(output_data)

def test_air( ):
    code = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
    expected_output = "Hello World!\n"
    program = VirtualMachine.compile(code)
    processor_table, instruction_table, memory_table, input_table, output_table = VirtualMachine.simulate(program)

    processor_table.test()
    instruction_table.test()
    memory_table.test()
    input_table.test()
    output_table.test()

    # extend tables, and re-test AIR
    challenges = [ExtensionField.main().sample(os.urandom(24)) for i in range(VirtualMachine.num_challenges())]

    processor_extension = VirtualMachine.extend_processor_table(processor_table, challenges)
    instruction_extension = VirtualMachine.extend_instruction_table(instruction_table, challenges)
    memory_extension = VirtualMachine.extend_memory_table(memory_table, challenges)
    input_extension = VirtualMachine.extend_input_table(input_table, challenges)
    output_extension = VirtualMachine.extend_output_table(output_table, challenges)

    processor_extension.test()
    instruction_extension.test()
    memory_extension.test()
    input_extension.test()
    output_extension.test()
