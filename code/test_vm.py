from vm import *

def test_vm():
    code = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
    expected_output = "Hello World!\n"
    #program = ">++++++++++[>+++><<-]>+++><<>."
    #expected_output = "!"
    input_data, output_data = VirtualMachine.execute(code)
    output_data = "".join(od for od in output_data)
    assert(output_data == expected_output), f"output data invalid; given:\"{output_data}\", but should be \"{expected_output}\""
    print(output_data)

def test_simulate():
    code = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
    expected_output = "Hello World!\n"
    #program = ">++++++++++[>+++><<-]>+++><<>."
    #expected_output = "!"
    program = VirtualMachine.compile(code)
    trace, output_data = VirtualMachine.simulate(program)
    output_data = "".join(od for od in output_data)
    assert(output_data == expected_output), f"output data invalid; given:\"{output_data}\", but should be \"{expected_output}\""
    print(output_data)
