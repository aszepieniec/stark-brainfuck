from vm import *

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
    output_data, trace, instruction_table, memory_table, input_table, output_table = VirtualMachine.simulate(program)

    processor_transition_constraints = VirtualMachine.processor_transition_constraints()
    processor_boundary_constraints = VirtualMachine.processor_boundary_constraints()

    for (register, cycle, value) in processor_boundary_constraints:
        assert(trace[cycle][register] == value), "processor boundary constraint not satisfied"

    for mpo in processor_transition_constraints:
        for clk in range(len(trace)-1):
            point = trace[clk] + trace[clk+1]
            assert(mpo.evaluate(point).is_zero()), "processor transition constraint not satisfied"

    instruction_transition_constraints = VirtualMachine.instruction_table_transition_constraints()
    instruction_boundary_constraints = VirtualMachine.instruction_table_boundary_constraints()

    for (column, row, value) in instruction_boundary_constraints:
        assert(instruction_table[row][column] == value), "instruction boundary constraint not satisfied"


    for itc in instruction_transition_constraints:
        for row in range(len(instruction_table)-1):
            point = instruction_table[row] + instruction_table[row+1]
            assert(itc.evaluate(point).is_zero()), f"instruction transition constraint {i} not satisfied in row {row} where point is {[p.value for p in point]}, and evaluation is {itc.evaluate(point).value}"
    
    memory_transition_constraints = VirtualMachine.memory_transition_constraints()
    memory_boundary_constraints = VirtualMachine.memory_boundary_constraints()

    print([m.value for m in memory_table[0]])

    for (column, row, value) in memory_boundary_constraints:
        assert(memory_table[row][column] == value), f"memory boundary constraint {(column, row, value.value)} not satisfied"

    for mtc in memory_transition_constraints:
        for row in range(len(memory_table)-1):
            point = memory_table[row] + memory_table[row+1]
            assert(mtc.evaluate(point).is_zero()), "memory transition constraint not satisfied"




