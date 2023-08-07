# BrainSTARK, Part V: Attack!

Surprise! There is an attack. Did you catch it? Don't worry; you're in good company if you didn't. The fact that attacks like this one can slip past our radars goes to show how incredibly subtle STARK engine design can be. It requires intense scrutiny if we want to have convincing proofs.

This epilogue to the BrianSTARK tutorial explains how the attack works, proposes a fix, and proves the correctness (soundness) of the fix.

## The Attack

Here's a Brainfuck program: `+><.-><+`. Clearly, this program outputs `1`. You don't need to run the VM for that – you can simulate it in your head.

Here is an execution trace for this program.

**Processor Table:**

| `clk` | `ip` | `ci` | `ni` | `mp` | `mv` | `inv` |
|-------|------|------|------|------|------|-------|
| 0 | 0 | `+` | `>` | 0 | 0 | 0 |
| 1 | 1 | `>` | `<` | 0 | 1 | 1 |
| 2 | 2 | `<` | `.` | 1 | 0 | 0 |
| 3 | 3 | `.` | `-` | 0 | **2** | **2** $^{-1}$ |
| 4 | 4 | `-` | `>` | 0 | 2 | 2 $^{-1}$ |
| 5 | 5 | `>` | `<` | 0 | 1 | 1 |
| 6 | 6 | `<` | `+` | 1 | 0 | 0 |
| 7 | 7 | `+` | 0   | 0 | 1 | 1 |
| 8 | 8 | 0   | 0   | 0 | 2 | 2 $^{-1}$ |

This execution trace illustrates the attack: it asserts that 2 is the program's output! Clearly something is wrong.

In an honest execution, when control returns to memory cell 0 in cycle 3, the value contained there should be what it was set to earlier, namely 1. However, this malicious trace sets the memory value to 2. Note that the Processor Table's AIR constraints do not constrain the value of `mv` when returning to a memory cell used prior – the correct value is instead established via a permutation argument with the Memory Table. Here it is.

**Memory Table:**

| `clk` | `mp` | `mv` |
|-------|------|------|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 5 | 0 | 1 |
| 7 | 0 | 1 |
| 8 | 0 | 2 |
| 3 | 0 | 2 |
| 4 | 0 | 2 |
| 2 | 1 | 0 |
| 6 | 1 | 0 |

Note that a) the rows of this table are a permutation of the rows of the Processor Table; and b) the AIR constraints of the Memory Table are satisfied. Indeed, these AIR constraints stipulate that
 - the first row consists of all zeros;
 - memory pointer can only increase by 1 or stay the same;
 - whenever memory pointer stays the same and there is a jump in clock cycle, then memory value has to remain the same.

The problem is that the clock cycle "jumps" from 8 to 3. This jump is downwards instead of upwards[^1], leading to an incorrect sorting. The correct sorting is never verified.

## Why Sorting Matters

Correct sorting means that the Memory Table is sorted by memory pointer first and by clock cycle second. If the Memory Table is sorted correctly, then two properties are ensured:
 1. For any given memory pointer, the list of rows with that memory pointer forms a *contiguous* sublist of the table. It is one chunk and not spread out over multiple locations and not interrupted by rows with another memory pointer.
 2. Within each region with the same memory pointer, the clock cycle is sorted in ascending order.

There may be other rules for structuring the Memory Table that ensure that these properties hold. The correct sorting rule certainly achieves it. These two properties, in combination with following AIR constraints, are sufficient to establish memory consistency.
 - Boundary constraint of the Memory Table: the first row consists of all zeros.
 - Transition constraint of the Memory Table: `mv` cannot change in the same row-pair as a jump in `clk`.
 - Transition constraint of the Memory Table: if `mp` changes, the `mv` in the next row must be zero.
 - Transition constraints of the Processor Table: `mv` cannot change with `mp` remaining the same unless `ci` $\in \lbrace$ `+`, `-` $\rbrace$.

To prove this implication, we must first define exactly what we mean by *memory consistency*. Intuitively, we mean that whenever control returns to a memory cell visited earlier, that cell contains the same value that was there when control left. More formally we can go with the following

**Definition 1 (memory consistency).** An execution trace $T$ of length $N+1$ has *memory consistency* if for every row $i$, the value of the indicated memory cell corresponds to when it was last set. Formally,

$ \forall i \in \lbrace 0, \ldots, N \rbrace : T[i][$ `mv` $] = T[j][$ `mv` $] $ for some $j \in \lbrace 0, \ldots, i-1 \rbrace$ satisfying both

 - $T[i][$ `mp` $] = T[j][$ `mp` $]$, and
 - $\forall k \in \lbrace j+1, \ldots i-1 \rbrace : T[k][$ `mp` $] \neq T[j][$ `mp` $] \, \vee \, T[k-1][$ `ci` $] \not \in \lbrace $ `+`, `-` $ \rbrace$;

or $T[i][$ `mv` $] = 0$ when such a $j$ does not exist.

This proposition has two layers of nested quantifiers. Note that the "for some" hides an $\exists$.

The claim is that the two properties listed above, in combination with the AIR constraints also listed above, are sufficient for memory consistency.

*Proof.*

Consider the rows as they appear in the Memory Table. In particular, the proposition $T[i][$ `mp` $] = T[j][$ `mp` $]$ filters for rows in the same contiguous region of constant `mp`.

When a $j$ satisfying the above description does not exist, then we distinguish two cases.
 1. Row `clk` = $i$ is the first row of a region of constant `mp`. In this case the transition constraint enforces that $T[i][$ `mv` $] = 0$.
 2. Row `clk` = $i$ is not the first row of a region of constant `mp` but none of the earlier rows in this region correspond to a `+` or `-` instruction. Let $l$ be the value of `clk` in the row before. Note that $T[l][$ `ci` $] \not \in \lbrace$ `+`, `-` $\rbrace$. Therefore if $l+1 = i$ then $T[l][$ `mv` $] = T[i][$ `mv` $]$ is enforced by the transition constraints of the Processor Table. Conversely if $l+1 \neq i$ then $T[l][$ `mv` $] = T[i][$ `mv` $]$ is enforced by the transition constraint of the Memory Table. By induction, the value of `mv` must be the same as that of the first row of the contiguous region, where the transition constraint enforces that it is zero, so $T[i][$ `mv` $] = 0$.

When a $j$ satisfying the above description does exist, the same induction argument lets us travel up in the contiguous region of the Memory Table until we hit row `clk` = $j$, at which point the same induction argument implies that $T[i][$ `mv` $] = T[j][$ `mv` $]$. This implication completes the proof. $\square$

## Solution

So the correct sorting is important, and the problem exploited by the attack was that it wasn't enforced. How to fix it?

The design space of fixes for this problem is large. The supporting code base elects for the following approach.

The Memory Table is extended as follows. Whenever there is a jump in `clk` by something other than 1, within a contiguous region of constant `mp`, insert dummy rows. Every dummy row increases `clk` by one, and repeats the previous row's values otherwise. The net effect is that all jumps within contiguous regions are erased. Next, add a column `dummy` whose values are `0` or `1` and whose purpose is to indicate which rows are dummy rows and which are not. 

The dummy rows should *not* be included in the permutation argument, and so the transition constraint that enforces the correct update of the running product should take it into account. Additionally, jumps in `clk` within the same contiguous region should be disallowed entirely. The changes give rise to the following AIR, presented here at the risk of repetition for the sake of standalone completeness.

The variables are $\mathsf{clk}, \mathsf{mp}, \mathsf{mv}, \mathsf{d}$; and additionally $\mathsf{clk}^\star, \mathsf{mp}^\star, \mathsf{mv}^\star, \mathsf{d}^\star$ for transition constraints. Furthermore, $\mathsf{ppa}$ and $\mathsf{ppa}^\star$ are the variables associated with the extension column that computes the running product for the permutation argument with the Processor Table, and $T_{\mathsf{ppa}}$ is the associated terminal value. The challenges are $d, e, f,$ and $\beta$.

The boundary constraints for the base table are: all columns start with zero: $\mathsf{clk}$, $\mathsf{mp}$, $\mathsf{mv}$, $\mathsf{d}$. The extension column is unconstrained in the first row, because this element should be the random initial.

The transition constraints for the base table are as follows.
 - Memory pointer increases by zero or by one: $(\mathsf{mp}^\star - \mathsf{mp}) \cdot (\mathsf{mp}^\star - \mathsf{mp} - 1)$.
 - In contiguous regions where the memory pointer remains the same, the clock cycle counter increases by one: $(\mathsf{mp}^\star - \mathsf{mp} - 1) \cdot (\mathsf{clk}^\star - \mathsf{clk} - 1)$.
 - If the memory pointer increases by one, then the new memory value must be zero: $(\mathsf{mp} - \mathsf{mp}^\star) \cdot \mathsf{mv}^\star$.
 - The dummy next is zero or one: $\mathsf{d}^\star \cdot (\mathsf{d}^\star - 1)$.
 - If the dummy value is set, the memory value cannot change: $\mathsf{d} \cdot (\mathsf{mv}^\star - \mathsf{mv})$.
 - If the dummy value is set, the memory pointer cannot change: $\mathsf{d} \cdot (\mathsf{mp}^\star - \mathsf{mp})$.

 As for the transition constraint for the extension column, the weighted sum of the current row is accumulated into the running product of the next row, but only if the current row's dummy variable is not set: $\mathsf{d} \cdot \mathsf{ppa} + (1 - \mathsf{d}) \cdot \mathsf{ppa} \cdot (d \cdot \mathsf{clk} + e \cdot \mathsf{mp} + f \cdot \mathsf{mv} - \beta) - \mathsf{ppa}^\star$.

 The terminal constraint is essentially the same as the transition constraint but with $\mathsf{ppa}^\star$ replaced by $T_{\mathsf{ppa}}$: $\mathsf{d} \cdot \mathsf{ppa} + (1 - \mathsf{d}) \cdot \mathsf{ppa} \cdot (d \cdot \mathsf{clk} + e \cdot \mathsf{mp} + f \cdot \mathsf{mv} - \beta) - T_{\mathsf{ppa}}$

 **Acknowledgements.** Many thanks to [Yuncong Zhang](https://github.com/yczhangsjtu) for drawing attention to the attack, and to [Thorkil](https://github.com/Sword-Smith) for demonstrating the attack, brainstorming for a solution, and helping to implement it.

| Back to Start: [Index](index) |
|-|
| [0](index) - [1](engine) - [2](brainfuck) - [3](arithmetization) - [4](next) - **5** |

[^1]: What does it even mean for a difference of prime field elements to go "upwards" or "downwards"? Finite field elements do not have a sign, after all. The point is that "upwards" and "downwards", as well as *sorting*, are defined relative to the evolution of the `clk` register.


