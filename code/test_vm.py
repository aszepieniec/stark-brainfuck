from extension import ExtensionField
from instruction_extension import InstructionExtension
from io_extension import IOExtension
from memory_extension import MemoryExtension
from processor_extension import ProcessorExtension
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
    a, b, c, d, e, f, alpha, beta, gamma, delta, eta = [ExtensionField.main().sample(os.urandom(24)) for i in range(VirtualMachine.num_challenges())]

    processor_extension = ProcessorExtension.extend(processor_table, a, b, c, d, e, f, alpha, beta, gamma, delta)
    instruction_extension = InstructionExtension.extend(instruction_table, a, b, c, alpha, eta)
    memory_extension = MemoryExtension.extend(memory_table, d, e, f, beta)
    input_extension = IOExtension.extend(input_table, gamma)
    output_extension = IOExtension.extend(output_table, delta)

    processor_extension.test()
    instruction_extension.test()
    memory_extension.test()
    input_extension.test()
    output_extension.test()
