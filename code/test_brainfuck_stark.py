from brainfuck_stark import *


def test_bfs():
    generator = BaseField.main().generator()
    xfield = ExtensionField.main()
    bfs = BrainfuckStark(generator, xfield)
    program = VirtualMachine.compile(">>[++-]<")
    processor_table, instruction_table, memory_table, input_table, output_table = bfs.vm.simulate(
        program)
    log_time = len(bin(len(processor_table.table))[2:])
    proof = bfs.prove(processor_table, instruction_table,
                      memory_table, input_table, output_table)
    verdict = bfs.verify(proof, log_time, program,
                         input_table.table, output_table.table)
    assert(verdict == True), "honest proof fails to verify"
