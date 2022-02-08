from concurrent.futures import process
from brainfuck_stark import *
from os.path import exists


def test_bfs():
    generator = BaseField.main().generator()
    xfield = ExtensionField.main()
    bfs = BrainfuckStark(generator, xfield)
    program = VirtualMachine.compile(">>[++-]<++++++++")
    program = VirtualMachine.compile(">++++++++++[>+++><<-]>+++><<>.")
    processor_table_table, instruction_table_table, memory_table_table, input_table_table, output_table_table = bfs.vm.simulate(
        program)
    running_time = len(processor_table_table)

    filename = "proof.dump"
    if exists(filename):
        fh = open(filename, "rb")
        proof = pickle.load(fh)
        fh.close()
    else:
        fh = open(filename, "wb")
        proof = bfs.prove(len(processor_table_table), program, processor_table_table, instruction_table_table,
                          memory_table_table, input_table_table, output_table_table)
        pickle.dump(proof, fh)
        fh.close()

    # proof = bfs.prove(running_time, program, processor_table_table, instruction_table_table,
    #                   memory_table_table, input_table_table, output_table_table)

    verdict = bfs.verify(proof, running_time, program,
                         input_table_table, output_table_table)
    assert(verdict == True), "honest proof fails to verify"
    print([ord(t.value) for t in output_table_table])
