# BrainSTARK, Part IV: Next Steps

Congratulations! If you made it to this point, you need to give yourself a pad on the back.

Wait! That's it – isn't there any code? But of course there is! There is a fully functional python implementation of the compiler, virtual machine, arithmetization, and STARK prover and verifier. Checkout the [github repository](https://github.com/aszepieniec/brainfuck-stark). You can use this source code
 - as clarification to some parts of the text that may remain unclear; or
 - as a reference implementation to compare your own implementation with; or
 - not at all, if you want to challenge yourself to write an implementation going only on the text; or
 - in practice, to generate and verify proofs for the correct execution of Brainfuck programs.

To run the test:
 ```
 $> git clone https://github.com/aszepieniec/brainfuck-stark.git
 $> cd brainfuck-stark/code/
 $> pypy3
 $>>>> from test_brainfuck_stark import *
 $>>>> test_bfs()
```

You might want to run it through pypy3 rather than python3 because it's faster.

## Next Steps

So you mastered the mathematics, the computer science, and the cryptography, involved in designing STARK engines. What's left to do? Here are a couple of suggestions.

### Performance

You may have noticed that the supporting implementation is really, *really* slow. Here are some things you might want improve if you want it all to work faster.

 1. **Hardcode AIR Polynomials.** The symbolic representation of the AIR polynomials – i.e., as multivariate polynomials – benefits reasoning and rapid prototyping, not to mention testing. However, in the end we only care about getting the value of the AIR polynomial as a function of one or two consecutive rows of a table. Depending on the specific constraint, there may be more efficient ways to compute this value than by evaluating a symbolic representation.

 2. **Reduce AIR Degree.** Asymptotically the bottleneck is the NTT step, which needed for computing the low-degree extension of codewords. This process transforms the codeword from one of length equal to the table it comes from, to length equal to the FRI evaluation domain length, which is much larger for two reasons: a) composition with the AIR polynomials generates quotient polynomials of large degree, and b) FRI works because the codeword corresponds to a polynomial of degree less $\rho$ times the length of the codeword, where $0 < \rho < 1$ is the expansion factor of FRI. While it is possible to set $\rho = 1/2$ in order to reduce the overhead of part (b), this assignment also generates the largest proofs. It is possible to gain much more by introducing new columns in order to reduce the degree of the AIR. In principle it is possible to reduce the degree of the AIR to two, although you might need to introduce a lot of columns for that.

 3. **Drop Python.** Python is excellent for short programs, for rapid prototyping, and for didactical purposes, but the truth is that it really bad when it comes to performance. A lot of performance can be gained by lifting the STARK engine to a language that operates closer to the hardware, like *e.g.* [Rust](https://www.rust-lang.org/).

### Security and Zero-Knowledge

For performance reasons, the security level was set to $\lambda = 2$. All the components are in place for higher security levels, so this is a straightforward change if you have a faster implementation.

Likewise, all the components are in place for producing (and verifying) *zero-knowledge* proofs, except Brainfuck does not naturally lend itself to programs that make use of this feature. Specifically, the program, the input, and the output, are all known to the verifier. What would it look like if you were to prove that you knew a secret input that, fed into the given program, generates the given output? It's only a small change away. Or you might want to modify the language to include, say, the operation `?` which provides one secret byte of input alongside the `,` operation which provides one public byte.

### Modify the Instruction Set Architecture

If we are going to add new operations, why not get rid of old ones too? You could go so far as to design a whole new instruction set architecture and then generate and verify STARK proofs for programs in that ISA. If you go down that path, here are some suggestions.
 - A stack machine with RAM is preferable to a register machine because the source and destination fields of an instruction in a register-based architecture are expensive to decode in AIRs. The stack is just another memory object whose consistency can be proven with another permutation argument.
 - Support nondeterminism natively. Consider introducing instructions that correctly guess the correct field element. Doing so makes the resulting STARK engine useful for a wide range of cryptographic protocols.
 - Arithmetic operations in the field $\mathbb{F}_p$ can be supported cheaply: $+, \times, -, /, x^{-1}$. These operations can be computed verifiably in one VM cycle. Using lookup tables it is possible to also support uint32 arithmetic, which is more compatible with existing computer hardware and serialization.
 - Speaking of lookup tables, it might be worthwhile to introduce dedicated tables for computing specific operations that are more complicated than simple arithmetic. For instance, hashes are needed all over the place in cryptographic protocols, so it makes sense to support that. And you might want to separate the AIR of the Processor Table from the AIR of the Hash Table.
 - If hashing is natively supported, you might as well optimize the virtual machine architecture for verifying STARK proofs, and then for proving the correct verification of STARK proofs. The result is a *recursive STARK*. Once recursion has been achieved, it is possible to do *incrementally verifiable computation (IVC)*, where you prove the correct verification of a STARK and one extra step of computation. This reduces the cost of producing a proof for very large computations from $O(N \log N)$ to $O(N)$, if you are willing to accept the qualitatively weaker security guarantee. Additionally, you can design *proof carrying data (PCD)* schemes, where by participants in general protocols build on top of each other's proofs.
 - Consider contributing to [Triton VM](https://github.com/TritonVM/triton-vm)! Okay, you got me. This entire tutorial was designed as a gateway drug to get you hooked on Triton. Seriously though, if you liked the subject matter in this tutorial then contributing to a STARK engine is the next logical step. And if you agree with these suggestions for designing a new VM for a STARK engine then you will be pleased to know that we took them to heart when designing Triton.

| (the end) |
|-|
| [0](index) - [1](engine) - [2](brainfuck) - [3](arithmetization) - **4** |
