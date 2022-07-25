# BrainSTARK, Part II: Brainfuck

## Brainfuck

[Brainfuck](https://en.wikipedia.org/wiki/Brainfuck) is an esoteric programming language consisting just of eight instructions. Despite its simplicity it is Turing-complete, meaning that it is capable of executing any algorithm. And while that simplicity makes producing practical programs rather challenging, it is also makes Brainfuck an excellent choice with which to illustrate the operational principles involved in, say, building a STARK engine.

### The Programming Language

Brainfuck defines an automaton that interacts with a program and a memory consisting of 30000 bytes initialized to zero. The state of the automaton consists of an instruction pointer, which points into the program; and a data pointer which points into the memory. There are 8 instructions that update the automaton's state, possibly with side-effects:
 - `[` jumps to the matching `]` instruction if the currently indicated memory cell is zero.
 - `]` jumps to the matching `[` if the currently indicated memory cell is nonzero.
 - `+` increments by one (modulo 256) the value of the currently indicated memory cell.
 - `-` decrements by one (modulo 256) the value of the currently indicated memory cell.
 - `<` decrements the memory pointer by one.
 - `>` increments the memory pointer by one.
 - `.` outputs the value of the currently indicated memory cell.
 - `,` reads a byte from the user input and stores it in the currently indicated memory cell.

### Arithmetic Brainfuck

In order to prove the correct execution of a Brainfuck program efficiently, it pays to define the following dialect. Since the modulus used is $p = 2^{64} - 2^{32} + 1$ it is convenient if the memory elements are elements of the field defined by this prime. Likewise for the instruction and data pointers. So in particular, $p$ is also the number of memory cells as well as the total number of instructions in a program.

### Brainfuck Assembler

Brainfuck is not an instruction set architecture because the instructions are not self-contained. They depend on context. Specifically, the `[` and `]` instructions refer to the locations of their matching partners.

To remedy this deficiency, modify the `[` and `]` instructions as follows. The field element immediately following `[` or `]` contains the destination address of the potential jump. The instruction set is now variable-size but nevertheless defines a machine architecture.

## Compiling Brianfuck PL to Brainfuck ASM

The difference between Brainfuck-programming-language and Brainfuck-assembler necessitates a compiler that computes the mapping. Fortunately, most advanced compiler construction tools are not necessary as there is an exceedingly simple [pushdown automaton](https://en.wikipedia.org/wiki/Pushdown_automaton) that achieves this translation task.

## Running Brainfuck
