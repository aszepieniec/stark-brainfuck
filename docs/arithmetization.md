# BrainSTARK, Part III: Arithmetization of Brainfuck VM

The virtual machine defines the evolution of two registers and memory. The generic [STARK engine](engine) already contains a high-level description of how memory might work. Therefore, let's focus for starters on the evolution of the set of registers in the processor.

Using two registers `ip` and `mp` for instruction pointer and memory pointer like the VM defines makes sense. However, this selection is too limited on its own. The constraints need to depend not just on the index contained in `ip` and `mp`, but also on the values these registers point to. To this end, introduce `mv` (*memory value*) and `ci` (*current instruction*). If the current instruction is a potential jump, then we also need to know where to jump to. This address is contained in the next instruction, and so a register is needed for that purpose: `ni`. Also, a potential jump requires some constraints be enforced if `mv` is nonzero and other constraints be enforced if `mv` is zero. The only way to enforce constraints conditioned on the zero-ness of some variable, is with an expression contains the inverse of this variable if it exists and 0 otherwise. Let `inv` be this register, and so in particular we have that `mv * inv` is always 0 or 1. Lastly, in order to make a permutation argument work for establishing correct memory accesses, it is necessary to keep track of jumps in a cycle counter in the table where the rows are sorted for memory address. To this end, introduce a register `clk` whose only purpose is to count the number of cycles have passed.

So for reference, this is the list of registers in our processor:
 - `clk` – clock, counts the number of cycles that have passed.
 - `ip` – instruction pointer, points to the current instruction in the program.
 - `ci` – current instruction, value of the program at `ip`.
 - `ni` – next instruction, value of the program at location `ip+1` or 0 if out of bounds.
 - `mp` – memory pointer, points to a memory cell.
 - `mv` – memory value, value of the memory pointed at by `mp`.
 - `inv` – inverse, takes the value 0 if `mv` is zero, or the multiplicative inverse of `mv` otherwise.

The processor does not evolve in isolation; rather, it receives an input and produces an output, and it interacts with a program and a memory. These interactions must also be authenticated. Concretely, this means there must be permutation and evaluation arguments between this table, the *Processor Table*, and other Tables.

Specifically:
 - The processor reads from and writes to memory. Whenever the memory pointer `mp` is reset to a prior value, the memory value `mv` must be consistent with the last time this memory cell was set. This consistency is enforced through a permutation argument with the *Memory Table*.
 - The processor reads instructions from a memory-like object that is schematically located in between the processor and the program. On the one hand, a permutation argument establishes that every tuple `(ip, ci, ni)` that the processor ever assumes has a matching row in this *Instruction Table*. On the other hand, an evaluation argument establishes that all rows of this Instruction Table correspond to an instruction and its successor at the given location in the program. The verifier, who has cleartext access to the program, evaluates this terminal locally.
 - The processor reads inputs from a stream of symbols called the Input Table. An evaluation argument establishes that the symbols read by the processor are identical to the symbols that make up the 'input' part of the computational integrity claim.
 - The processor writes outputs to another stream of symbols called the Output Table. An evaluation argument establishes that the symbols written by the processor are identical to the symbols that make up the 'output' part of the computational integrity claim.

This description gives rise to the following diagram representation of the various tables and their interactions. The red arrows denote evaluation arguments; the blue arrows denote permutation arguments.

![](graphics/table-relations.svg)

One feature of this diagram might be confusing. If the verifier has cleartext access to the input and the output, then surely he can compute the evaluation terminals locally without bothering with the InputTable and OutputTable? That observation is entirely correct. However, the motivation for the present architecture is to enable extensions. For instance, a natural extension is for the input to remain secret, or even for the same secret input to be reused in different places within the same proof system, or even across different proofs. For fancier constructions like these, an explicit InputTable comes in handy.

## Tables

## Base Tables

## Table Extensions
