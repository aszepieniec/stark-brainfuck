from concurrent.futures import process
from brainfuck_stark import *
from os.path import exists


def test_bfs():
    generator = BaseField.main().generator()
    xfield = ExtensionField.main()
    bfs = BrainfuckStark(generator, xfield)
    program = VirtualMachine.compile(">>[++-]<+++")
    processor_table_table, instruction_table_table, memory_table_table, input_table_table, output_table_table = bfs.vm.simulate(
        program)
    # log_time = len(bin(len(processor_table.table)-1)[2:])
    # print("lengh of processor table:", len(processor_table.table))
    # print("log time:", log_time)

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

    verdict = bfs.verify(proof, len(processor_table_table), program,
                         input_table_table, output_table_table)
    assert(verdict == True), "honest proof fails to verify"
    print("\o/")
