Notes for Alan's Brainfuck STARK tutorial
- execution starts in the `test_brainfuck_stark` file where a program can be declared
- First the program is compiled to a list of assembly instructions using `compile`, this compiles Brainfuck
  jump instructions into jump instructions with absolute addresses.
- `simulate` returns matrices which are generalizations of execution traces. The processor matrix is a regular 2-dimensional
  trace where the column index defines the register and the row index defines the cycle. The `matrix` values go into a table
  object with an associated name. The table names are: "processor table", "instruction table", "memory table", "input table",
  and "output table". Simulate returns the matrices which form the basis for populating the tables.
- Each table serves a purpose in the STARK engine: the processor table handles basic consistency in the execution trace,
  ensuring that the cycle count register increments its value by one in each cycle; the instruction table ensures that the
  program is consistent with the instructions that are executed on the virtual machine; the memory table ensure consistency
  in the memory of the Brainfuck virtual machine; the input and output tables ensure that the program is consistent with its
  user input and the values that the it prints to standard out, respectively.
- `length` of a table defines the number of rows before padding
- `height` of a table defines the number of rows after padding to the nearest power of 2.
- Changes to FRI: 
  - The FRI does not output indices anymore. This is done in anticipation of HALO-style recursion.
  - FRI has a new "Domain" object with some methods for evaluation and interpolation.
  
- `security_level` refers to the log2 of the number of calculations required to attack the protocol with a false proof
- The number of randomizers should probably be 2 * security_level
- Each table has its own omicron, this imocron is calculated from the smooth generator, which is the generator that forms
  the group of of the highest order which is a power of 2 for this prime field (the B field)
- All table objects are stored in the "base tables" field.
- Permutation arguments make statements about set equality, that a list is a permutation of another list.
- This the base table columns are interpolated, then the extension columns are calculated from the base columns and from
  Fiat-shamir randomness (verifier randomness).
- "terminal value": the last value of a permutation running product, enforced through terminal constraints.
- Base columns can be calculated as the program runs. Extension columns require verifier input to be calculated.
- AIR constraints refer to relations between both extension columns and base columns. The extension columns are introduced
  in order for us to arithmetize (express as polynomials) a wider range of constraints than we would be able to, had we only
  had the base columns. Using extension columns we can for example ensure that if a read-only memory address is read from
  multiple times in the program, we always get the same value back. This constraint is necessary to prove the integral
  evaluation of a program, and cannot be achieved without adding more columns than the running simulation of the program
  can achieve.

