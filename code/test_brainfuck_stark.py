from brainfuck_stark import *


def test_sanity():
    bfs = BrainfuckStark()
    program = VirtualMachine.compile(">>[++-]<")
    processor_table, instruction_table, memory_table, input_table, output_table = bfs.vm.simulate(
        program)
    bfs.prove(processor_table, instruction_table,
              memory_table, input_table, output_table)
