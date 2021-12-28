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


