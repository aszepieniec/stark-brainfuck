from concurrent.futures import process
from brainfuck_stark import *
from os.path import exists


def test_bfs():
    generator = BaseField.main().generator()
    xfield = ExtensionField.main()
    program = VirtualMachine.compile(">>[++-]<++++++++")
    program = VirtualMachine.compile(
        ">++++++++++[>+++><<-]>+++><<>.................")
    program = VirtualMachine.compile(
        ",+.")

    running_time, input_symbols, output_symbols = VirtualMachine.run(program)

    print("running time:", running_time)
    print("input symbols:", input_symbols)
    print("output_symbols:", output_symbols)

    bfs = BrainfuckStark(running_time, program, input_symbols, output_symbols)

    # Print "Hello World!"
    # program = VirtualMachine.compile(
    #     "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.")
    processor_matrix, instruction_matrix, input_matrix, output_matrix = bfs.vm.simulate(
        program, input_data=input_symbols)
    running_time = len(processor_matrix)

    filename = "proof.dump"
    if exists(filename):
        fh = open(filename, "rb")
        proof = pickle.load(fh)
        fh.close()
    else:
        proof = bfs.prove(len(processor_matrix), program, processor_matrix,
                          instruction_matrix, input_matrix, output_matrix)
        fh = open(filename, "wb")
        pickle.dump(proof, fh)
        fh.close()

    # proof = bfs.prove(running_time, program, processor_matrix, instruction_matrix,
    #                   memory_matrix, input_matrix, output_matrix)

    # collapse matrix into list for input and output
    input_symbols = [row[0] for row in input_matrix]
    output_symbols = [row[0] for row in output_matrix]

    verdict = bfs.verify(proof)
    assert(verdict == True), "honest proof fails to verify"
    print("output length was:", len(output_symbols))
    print("proof verified with output: \"" + "".join(
        [chr(t.value) for t in output_symbols]) + "\"")
