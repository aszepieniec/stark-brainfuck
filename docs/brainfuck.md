# BrainSTARK, Part II: Brainfuck

## Brainfuck

[Brainfuck](https://en.wikipedia.org/wiki/Brainfuck) is an esoteric programming language consisting of just eight instructions. Despite its simplicity it is Turing-complete, meaning that it is capable of executing any algorithm. While its simplicity makes producing practical programs rather challenging, it also makes Brainfuck an excellent choice with which to illustrate the operational principles involved in, say, building a STARK engine.

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

The wrap-around modulo 256 is just one reading of the specification. It is rather undefined about what happens when incrementing a cell whose value is already 255 or decrementing a cell whose value is 0.

### Prime Field Brainfuck

In order to prove the correct execution of a Brainfuck program efficiently, it pays to define the following dialect. Since the modulus used is $p = 2^{64} - 2^{32} + 1$ it is convenient if the memory elements are elements of the field defined by this prime. Likewise for the instruction and data pointers. So in particular, $p$ is also the number of memory cells as well as the maximum total number of instructions in a program.

### Brainfuck ISA

Brainfuck is not an instruction set architecture (ISA) because the instructions are not self-contained. They depend on context. Specifically, the `[` and `]` instructions refer to the locations of their matching partners.

To remedy this deficiency, modify the `[` and `]` instructions as follows. The field element immediately following `[` or `]` contains the destination address of the potential jump. The instruction set is now variable-size but nevertheless defines a machine architecture.

## Compiling Brianfuck Programs to Brainfuck Assembler

The difference between Brainfuck-programming-language and Brainfuck-assembler necessitates a compiler that computes the mapping. Fortunately, most advanced compiler construction tools are not necessary as there is an exceedingly simple [pushdown automaton](https://en.wikipedia.org/wiki/Pushdown_automaton) that achieves this translation.

Pushdown automata are generally the computational model of choice for building parsers for programming languages. These parsers output [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree)s, which are then transformed by the next step in the compilation pipeline. In the present case, however, desired output is another sequence of symbols that is very similar to the input sequence and it is possible to by-pass the abstract syntax tree altogether.

The compiler presented here is *almost* streaming, meaning that it runs over the input sequence once and starts outputting symbols before it reaches the end. The qualifier "almost" indicates that occasionally, the compiler will output placeholder symbols whose concrete value will be set later. 

The compiler keeps track of a stack that stores the locations (in the output sequence) of the `[` symbols that have not yet been closed by a matching `]` symbol. Let `c` denote a counter that tracks the total number of symbols sent to output so far. 
 - Whenever a `[` symbol is encountered, two things happen: a) `c` is pushed to the stack, and b) two symbols are pushed to the output: `[` and the placeholder `*`.
 - Whenever a `]` symbol is encountered, three things happen: a) the location `i` of the matching `[` is read from the stack and popped, b) the placeholder in the output sequence at location `i+1` is set to `c+2`, and c) two symbols are pushed to the output: `]` and `i+2`.
 - Whenever any other symbol is encountered, it is pushed to the output sequence with no changes to the stack.

## Brainfuck VM

The state of a brainfuck virtual machine consists of two registers and the memory. The registers are the instruction pointer `ip` and the data pointer `dp`; both are initially set to zero. The memory can be represented by various data structures but if it is desirable to avoid allocating unnecessary memory on the host machine, then a dictionary is a good choice. When the dictionary is queried for the value of a new key, it returns 0 -- consistent with the initial value of memory cells.

Any virtual machine defines a *state transition function*. Running the virtual machine consists of repeatedly applying this function to the state, until the termination criterion is met.

In the case of the Brainfuck VM, the program terminates when the instruction pointer points beyond the length of the program, or in pseudocode when `ip >= len(program)`. The state transition function depends on the instruction. Let `program[i]` denote the instruction at location `i` and let `data[i]` denote the memory cell at location `i`.
 - If `program[ip] == '['` then a) if `data[dp] == 0` jump, *i.e.*, set `ip = program[ip+1]`, or b) if `data[dp] =/= 0` then skip past the destination address by setting `ip = ip + 2`.
 - If `program[ip] == ']'` then a) if `data[dp] == 0` then skip past the destination address by setting `ip = ip + 2`, or b) if `data[dp] =/= 0` then jump, *i.e.*, set `ip = program[ip+1]`.
 - If `program[ip] == '<'` then a) decrement the data pointer `dp = dp - 1` and b) increment the instruction pointer `ip = ip + 1`.
 - If `program[ip] == '>'` then a) increment the data pointer `dp = dp + 1` and b) increment the instruction pointer `ip = ip + 1`.
 - If `program[ip] == '+'` then a) increment the indicated data element `data[dp] = data[dp] + 1` and b) increment the instruction pointer `ip = ip + 1`.
 - If `program[ip] == '-'` then a) decrement the indicated data element `data[dp] = data[dp] - 1` and b) increment the instruction pointer `ip = ip + 1`.
 - If `program[ip] == '.'` then a) output the indicated data element `data[dp]` and b) increment the instruction pointer `ip = ip + 1`.
 - If `program[ip] == ','` then a) set the indicated data element `data[dp]` to the input symbol and b) increment the instruction pointer `ip = ip + 1`.


| Next up: [Part III: Arithmetization](arithmetization) |
|-|
| [0](index) - [1](engine) - **2** - [3](arithmetization) - [4](next) - [5](attack) |