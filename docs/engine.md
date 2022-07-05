# BrainSTARK, Part I: STARK Engine

## STARK Recap

### AIR and AET

At the heart of a STARK is the Algebraic Execution Trace (AET), which is a table of $\mathsf{w}$ columns and $T+1$ rows. Every column tracks the value of one register across time. Every row represents the state of the entire machine at one point in time. There are $T+1$ rows because the machine was initialized to the initial state before it was run for $T$ time steps.

The AET is *integral* if all Arithmetic Intermediate Representation (AIR) constraints are satisfied. These constraints come in several classes.

 - Boundary constraints apply only at the beginning or at the end. Boundary constraints are polynomials in $\mathsf{w}$ variables. Boundary constraints enforce, for instance, that the initial state is correct.
 - Transition constraints apply to every consecutive pair of rows, and are there polynomials in $2\mathsf{w}$ variables. They enforce that the state transition function was evaluated correctly.

The prover runs a polynomial interpolation subprocedure to find, for every column, the unique polynomial of degree $T$ that takes the value $i$ rows down in the point $\omicron^i$, where $\omicron$ is a generator of a subgroup of order $T+1$. These polynomials are called the trace polynomials.

Evaluating the AIR constraints gives rise to *boundary* and *transition polynomials*. Moreover, every AIR constraint defines a support domain $D \subset \langle \omicron \rangle$ of points where it applies, and with it a zerofier $Z(X)$, which is the unique monic polynomial of degree $|D|$ that evaluates to zero on all of $D$ and no-where else. If the AET satisfies the AIR constraint then this zerofier divides the boundary or transition polynomial cleanly; if the AET does not satisfy the AIR constraint then this division has a nonzero remainder.

The prover continues with the *quotient polynomials* of the previous step. Specifically, he wishes to establish their bounded degree. If the AET satisfies the AIR constraints, then the quotient polynomials will be of low degree. If the AET does not satisfy the AIR constraints the malicious prover might be able to find impostor polynomials that agree with the division relation in enough points, but the point is that this impostor polynomial will necessarily have a *high* degree. And then the prover will fail to prove that its degree is low.

The prover evaluates all these polynomials on a coset of the subgroup spanned by $\Omega$, whose order is $N > T+1$. This process of evaluation gives rise to Reed-Solomon codewords of length $N$. The next step is to zip these codewords into one vector of length $N$ of tuples of finite field elements. And then this vector's Merkle tree is computed. The root is sent the verifier.

### FRI

For every quotient polynomial $q_i(X)$, there is a degree bound $b_i$ originating from the known trace length $T$ and AIR constraint degree. The prover combines a

## Tables

### Permutation Arguments

### Evaluation Arguments

## STARK Engine Workflow

