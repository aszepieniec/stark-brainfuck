# BrainSTARK, Part I: STARK Engine

## STARK Recap

The word STARK can mean one of several things.

 - STARK $_1$: The acronym stands for Scalable Transparent ARgument of Knowledge, which applies to any SNARK (or even interactive SARK) has at most a polylogarithmic prover overhead and no trusted setup.
 - STARK $_2$: A specific way of building STARK $_1$'s by using AETs and AIRs and plugging the resulting polynomials into FRI to prove their bounded degree.
 - STARK $_3$: a concrete proof object resulting from a STARK $_2$ prover.

This short review focuses on STARK $_2$.

### AIR and AET

At the heart of a STARK is the Algebraic Execution Trace (AET), which is a table of $\mathsf{w}$ columns and $T+1$ rows. Every column tracks the value of one register across time. Every row represents the state of the entire machine at one point in time. There are $T+1$ rows because the machine was initialized to the initial state before it was run for $T$ time steps.

The AET is *integral* if all Arithmetic Intermediate Representation (AIR) constraints are satisfied. These constraints come in several classes.

 - Boundary constraints apply only at the beginning or at the end. Boundary constraints are represented as polynomials in $\mathsf{w}$ variables. Boundary constraints enforce, for instance, that the initial state is correct.
 - Transition constraints apply to every consecutive pair of rows, and can be represented as polynomials in $2\mathsf{w}$ variables. They enforce that the state transition function was evaluated correctly.

While the AIR constraints can be represented as multivariate polynomials, it helps to think of them as maps $\mathbb{F}^{\mathsf{w}} \rightarrow \mathbb{F}$ or $\mathbb{F}^{2\mathsf{w}} \rightarrow \mathbb{F}$. This perspective extends the evaluation map from vectors of $\mathsf{w}$ or $2\mathsf{w}$ field elements to vectors of as many codewords or polynomials.

The prover runs a polynomial interpolation subprocedure to find, for every column, the unique polynomial of degree $T$ that takes the value $i$ rows down in the point $\omicron^i$, where $\omicron$ is a generator of a subgroup of order $T+1$. These polynomials are called the trace polynomials.

Evaluating the AIR constraints in the trace polynomials gives rise to *boundary* and *transition polynomials*. Moreover, every AIR constraint defines a support domain $D \subset \langle \omicron \rangle$ of points where it applies, and with it a zerofier $Z(X)$, which is the unique monic polynomial of degree $|D|$ that evaluates to zero on all of $D$ and no-where else. If the AET satisfies the AIR constraint then this zerofier divides the boundary or transition polynomial cleanly; if the AET does not satisfy the AIR constraint then this division has a nonzero remainder.

The prover continues with the *quotient polynomials* of the previous step. Specifically, he wishes to establish their bounded degree. If the AET satisfies the AIR constraints, then the quotient polynomials will be of low degree. If the AET does not satisfy the AIR constraints the malicious prover might be able to find impostor polynomials that agree with the division relation in enough points, but the point is that this impostor polynomial will necessarily have a *high* degree. And then the prover will fail to prove that its degree is low.

### Commitment

The prover commits to these polynomials as follows. First he evaluates them on a coset of the subgroup spanned by $\Omega$, whose order is $N > T+1$. This process of evaluation gives rise to Reed-Solomon codewords of length $N$. Next, a Merkle tree of this codeword is computed. The root is the commitment to the polynomial, and it is sent to the verifier.

One obvious optimization is available at this point. It is possible to zip the codewords before computing Merkle trees. In fact, after zipping, only one Merkle tree needs to be computed. The leafs of this Merkle tree correspond to tuples of field elements.

The next step is to combine the polynomials' codewords into one codeword using random weights from the verifier. For every quotient polynomial $q_i(X)$, there is a degree bound $b_i$ originating from the known trace length $T$ and AIR constraint degree. The prover combines the nonlinear combination
$$ \sum_{i=0} \alpha_i \cdot q_i(X) + \beta_i \cdot X^{\mathsf{d} - b_i} \cdot q_i(X) \enspace ,$$
where the weights $\alpha_i$ and $\beta_i$ are provided by the verifier, and where $\mathsf{d}$ is the maximum degree bound provably be FRI. The codeword associated with this polynomial is the input to FRI.

### FRI

FRI establishes that the input codeword has a low-degree defining polynomial. It does this by folding the working codeword in on itself using a random weight supplied by the verifier, over the course of several rounds. This folding procedure sends low-degree codewords to low-degree codewords, and high-degree codewords to high-degree codewords. The verifier checks the correct folding relation by inspecting the codewords in a series of indices.

Rather than transmitting the codewords in the clear, the prover first compresses them using a Merkle tree, and then transmits only the root. After all the Merkle roots are in, the verifier announces the indices where he would like to inspect the committed codewords. The prover answers by sending the indicated leafs along with their authentication paths.

In the last round, the prover sends the codeword in the clear. The length of this codeword is what happens after $\mathsf{r}$ halvings – specifically, its length is $N/2^{\mathsf{r}}$, where $N$ as the length of the original codeword.

## From STARK to STARK Engine

The STARK mechanics described above suffice for proving the integral evolution of a *simple* state, *i.e.,* on that is fully determined by $\mathsf{w}$ registers. It suffices for a digital signature scheme based on a proof of knowledge of a preimage under a hash function, or a verifiable delay function. But there is a considerable gap between that and a general-purpose virtual machine.

For instance, a machine following the [von Neumann architecture](https://en.wikipedia.org/wiki/Von_Neumann_architecture) needs to
 1. read the next instruction from memory and decode it;
 2. (possibly) read a word from memory or standard input;
 3. update the register set in accordance with the instruction;
 4. (possibly) write a word to memory or standard output.

 ![Von Neumann machine architecture](graphics/von-neumann.svg)

At best, the simple state evolution descibes the evolution of the machine's register set. But how does it interact with external data sources and sinks? More importantly, how to prove and verify the integrity of these interactions?

## Tables

### Permutation Arguments

### Evaluation Arguments

## Differences with respect to the Anatomy

### Salting the Leafs

When zero-knowledge is important, the authentication paths in the Merkle tree of the zipped codeword leak small amounts of information  this step involves appending raw randomness to every leaf before computing the Merkle tree. With this option enabled, an authentication path for one leaf leaks almost no information about the leaf's sibling -- and exactly zero bits of information in the random oracle model, which is when an idealized hash function is used for the security proof.

### Cleaner FRI Interface

## STARK Engine Workflow

