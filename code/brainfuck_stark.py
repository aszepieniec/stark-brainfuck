from typing import List
from evaluation_argument import EvaluationArgument, ProgramEvaluationArgument
from extension_field import ExtensionField, ExtensionFieldElement
from ip import ProofStream
from merkle import Merkle
from fri import *
from instruction_table import InstructionTable
from io_table import InputTable, OutputTable
from memory_table import MemoryTable
from permutation_argument import PermutationArgument
from processor_table import ProcessorTable
from salted_merkle import SaltedMerkle
from univariate import *
from multivariate import *
from ntt import *
from functools import reduce
import os


class BrainfuckStark:
    field = BaseField.main()
    xfield = ExtensionField.main()

    def __init__(self, running_time, memory_length, program, input_symbols, output_symbols):
        # set fields of computational integrity claim
        self.running_time = running_time
        self.memory_length = memory_length
        self.program = program
        self.input_symbols = input_symbols
        self.output_symbols = output_symbols

        # set parameters
        log_expansion_factor = 4  # for compactness
        log_expansion_factor = 2  # for speed
        self.expansion_factor = 1 << log_expansion_factor
        self.security_level = 160  # for security
        self.security_level = 2  # for speed
        self.num_colinearity_checks = self.security_level // log_expansion_factor
        assert (self.expansion_factor & (self.expansion_factor - 1)
                == 0), "expansion factor must be a power of 2"
        assert (self.expansion_factor >=
                4), "expansion factor must be 4 or greater"
        assert (self.num_colinearity_checks * len(bin(self.expansion_factor)
                                                  [3:]) >= self.security_level), "number of colinearity checks times log of expansion factor must be at least security level"

        self.num_randomizers = 1  # TODO: self.security_level

        # instantiate table objects
        order = 1 << 32
        smooth_generator = BrainfuckStark.field.primitive_nth_root(order)

        self.processor_table = ProcessorTable(
            self.field, running_time, self.num_randomizers, smooth_generator, order)
        self.instruction_table = InstructionTable(
            self.field, running_time + len(program), self.num_randomizers, smooth_generator, order)
        self.memory_table = MemoryTable(
            self.field, memory_length, self.num_randomizers, smooth_generator, order)
        self.input_table = InputTable(
            self.field, len(input_symbols), smooth_generator, order)
        self.output_table = OutputTable(
            self.field, len(output_symbols), smooth_generator, order)

        self.tables = [self.processor_table, self.instruction_table,
                       self.memory_table, self.input_table, self.output_table]

        # instantiate permutation objects
        processor_instruction_permutation = PermutationArgument(
            self.tables, (0, ProcessorTable.instruction_permutation), (1, InstructionTable.permutation))
        processor_memory_permutation = PermutationArgument(
            self.tables, (0, ProcessorTable.memory_permutation), (2, MemoryTable.permutation))
        self.permutation_arguments = [
            processor_instruction_permutation, processor_memory_permutation]

        # instantiate evaluation objects
        input_evaluation = EvaluationArgument(
            8, 2, [BaseFieldElement(ord(i), self.field) for i in input_symbols])
        output_evaluation = EvaluationArgument(
            9, 3, [BaseFieldElement(ord(o), self.field) for o in output_symbols])
        program_evaluation = ProgramEvaluationArgument(
            [0, 1, 2, 10], 4, program)
        self.evaluation_arguments = [
            input_evaluation, output_evaluation, program_evaluation]

        # compute fri domain length
        self.max_degree = 1
        for table in self.tables:
            # Using one() here might lead to syzygies in weird edge cases,
            # but that shouldn't be the case for Brainfuck though.
            for air in table.transition_constraints_ext([self.xfield.one()] * 11):
                degree_bounds = [table.interpolant_degree()] * \
                    table.full_width * 2
                degree = air.symbolic_degree_bound(
                    degree_bounds) - (table.height - 1)
                if self.max_degree < degree:
                    self.max_degree = degree

        self.max_degree = BrainfuckStark.roundup_npo2(self.max_degree) - 1
        fri_domain_length = (self.max_degree+1) * self.expansion_factor

        # instantiate self.fri object
        generator = BrainfuckStark.field.generator()
        omega = BrainfuckStark.field.primitive_nth_root(fri_domain_length)
        self.fri = Fri(generator, omega, fri_domain_length,
                       self.expansion_factor, self.num_colinearity_checks, self.xfield)

    def get_terminals(self) -> List[ExtensionFieldElement]:
        terminals = [self.processor_table.instruction_permutation_terminal,
                     self.processor_table.memory_permutation_terminal,
                     self.processor_table.input_evaluation_terminal,
                     self.processor_table.output_evaluation_terminal,
                     self.instruction_table.evaluation_terminal]
        return terminals

    def sample_weights(self, number, randomness):
        return [self.xfield.sample(blake2b(randomness + bytes(i)).digest()) for i in range(number)]

    @staticmethod
    def sample_indices(number, randomness, bound):
        indices = []
        for i in range(number):
            byte_array = blake2b(randomness + bytes(i)).digest()
            integer = 0
            for b in byte_array:
                integer = integer * 256 + int(b)
            indices += [integer % bound]
        return indices

    @staticmethod
    def roundup_npo2(integer):
        if integer == 0 or integer == 1:
            return 1
        return 1 << (len(bin(integer-1)[2:]))

    def prove(self, program, processor_matrix, memory_matrix, instruction_matrix, input_matrix, output_matrix, proof_stream=None):
        running_time = len(processor_matrix)
        assert (running_time + len(program) == len(instruction_matrix))

        # populate tables' matrices
        self.processor_table.matrix = processor_matrix
        self.memory_table.matrix = memory_matrix
        self.instruction_table.matrix = instruction_matrix
        self.input_table.matrix = input_matrix
        self.output_table.matrix = output_matrix

        # pad table to height 2^k
        self.processor_table.pad()
        self.memory_table.pad()
        self.instruction_table.pad()
        self.input_table.pad()
        self.output_table.pad()

        # create proof stream if we don't have it already
        if proof_stream == None:
            proof_stream = ProofStream()

        # compute root of unity of large enough order
        # for fast (NTT-based) polynomial arithmetic
        omega = self.fri.domain.omega
        order = self.fri.domain.length
        while order > self.max_degree+1:
            omega = omega ^ 2
            order = order // 2

        randomizer_codewords = []
        randomizer_polynomial = Polynomial([self.xfield.sample(os.urandom(
            3*9)) for i in range(self.max_degree+1)])
        randomizer_codeword = self.fri.domain.xevaluate(
            randomizer_polynomial)
        randomizer_codewords += [randomizer_codeword]

        base_codewords = reduce(
            lambda x, y: x+y, [table.lde(self.fri.domain) for table in self.tables], [])
        all_base_codewords = randomizer_codewords + base_codewords

        base_degree_bounds = reduce(
            lambda x, y: x+y, [[table.interpolant_degree()] * table.base_width for table in self.tables], [])

        zipped_codeword = list(zip(*all_base_codewords))
        base_tree = SaltedMerkle(zipped_codeword)
        proof_stream.push(base_tree.root())

        # get coefficients for table extensions
        challenges = self.sample_weights(
            11, proof_stream.prover_fiat_shamir())

        initials = [self.xfield.sample(os.urandom(3*8))
                    for i in range(len(self.permutation_arguments))]

        for table in self.tables:
            table.extend(challenges, initials)

        terminals = self.get_terminals()

        extension_codewords = reduce(
            lambda x, y: x+y, [table.ldex(self.fri.domain, self.xfield) for table in self.tables], [])

        zipped_extension_codeword = list(zip(*extension_codewords))
        extension_tree = SaltedMerkle(zipped_extension_codeword)
        proof_stream.push(extension_tree.root())

        extension_degree_bounds = reduce(lambda x, y: x+y, [[table.interpolant_degree()] * (
            table.full_width - table.base_width) for table in self.tables], [])

        quotient_codewords = []

        for table in self.tables:
            quotient_codewords += table.all_quotients(
                self.fri.domain, table.codewords, challenges, terminals)

        quotient_degree_bounds = []
        for table in self.tables:
            quotient_degree_bounds += table.all_quotient_degree_bounds(
                challenges, terminals)

        # ... and equal initial values
        for pa in self.permutation_arguments:
            quotient_codewords += [pa.quotient(self.fri.domain)]
            quotient_degree_bounds += [pa.quotient_degree_bound()]

        for t in terminals:
            proof_stream.push(t)

        # get weights for nonlinear combination
        #  - 1 for randomizer polynomials
        #  - 2 for every other polynomial (base, extension, quotients)
        num_base_polynomials = sum(
            table.base_width for table in self.tables)
        num_extension_polynomials = sum(
            table.full_width - table.base_width for table in self.tables)
        num_randomizer_polynomials = 1
        num_quotient_polynomials = len(quotient_degree_bounds)
        weights_seed = proof_stream.prover_fiat_shamir()
        weights = self.sample_weights(
            num_randomizer_polynomials
            + 2 * (num_base_polynomials +
                   num_extension_polynomials +
                   num_quotient_polynomials),
            weights_seed)

        # compute terms of nonlinear combination polynomial
        terms = [randomizer_codeword]
        # base_codewords = processor_base_codewords + instruction_base_codewords + \
        # memory_base_codewords + input_base_codewords + output_base_codewords
        assert (len(base_codewords) ==
                num_base_polynomials), f"number of base codewords {len(base_codewords)} codewords =/= number of base polynomials {num_base_polynomials}!"
        for i in range(len(base_codewords)):
            terms += [[self.xfield.lift(c) for c in base_codewords[i]]]
            shift = self.max_degree - base_degree_bounds[i]
            terms += [[self.xfield.lift((self.fri.domain(j) ^ shift) * base_codewords[i][j])
                      for j in range(self.fri.domain.length)]]
            if os.environ.get('DEBUG') is not None:
                print(f"before domain interpolation")
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                print(
                    f"degree of interpolation, base_codewords({i}): {interpolated.degree()}")
                assert (interpolated.degree() <= self.max_degree)
        assert (len(extension_codewords) ==
                num_extension_polynomials), f"number of extension codewords {len(extension_codewords)} =/= number of extension polynomials {num_extension_polynomials}"
        for i in range(len(extension_codewords)):
            terms += [extension_codewords[i]]
            shift = self.max_degree - extension_degree_bounds[i]
            terms += [[self.xfield.lift(self.fri.domain(j) ^ shift) * extension_codewords[i][j]
                      for j in range(self.fri.domain.length)]]
            if os.environ.get('DEBUG') is not None:
                print(f"before domain interpolation")
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                print(
                    f"degree of interpolation, extension_codewords({i}): {interpolated.degree()}")
                assert (interpolated.degree() <= self.max_degree)
        assert (len(quotient_codewords) ==
                num_quotient_polynomials), f"number of quotient codewords {len(quotient_codewords)} =/= number of quotient polynomials {num_quotient_polynomials}"

        for quotient_codeword, quotient_degree_bound in zip(quotient_codewords, quotient_degree_bounds):
            terms += [quotient_codeword]
            if os.environ.get('DEBUG') is not None:
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                assert (interpolated.degree() == -1 or interpolated.degree() <=
                        quotient_degree_bound), f"for unshifted quotient polynomial {i}, interpolated degree is {interpolated.degree()} but > degree bound i = {quotient_degree_bound}"
            shift = self.max_degree - quotient_degree_bound

            terms += [[self.xfield.lift(self.fri.domain(j) ^ shift) * quotient_codeword[j]
                      for j in range(self.fri.domain.length)]]
            if os.environ.get('DEBUG') is not None:
                print(f"before domain interpolation")
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                print(
                    f"degree of interpolation, , quotient_codewords({i}): {interpolated.degree()}")
                print("quotient  degree bound:", quotient_degree_bound)
                assert (interpolated.degree(
                ) == -1 or interpolated.degree() <= self.max_degree), f"for (shifted) quotient polynomial {i}, interpolated degree is {interpolated.degree()} but > max_degree = {self.max_degree}"

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for i)
        assert (len(terms) == len(
            weights)), f"number of terms {len(terms)} is not equal to number of weights {len(weights)}"

        combination_codeword = reduce(
            lambda lhs, rhs: [l+r for l, r in zip(lhs, rhs)], [[w * e for e in t] for w, t in zip(weights, terms)], [self.xfield.zero()] * self.fri.domain.length)

        # commit to combination codeword
        combination_tree = Merkle(combination_codeword)
        proof_stream.push(combination_tree.root())

        # get indices of leafs to prove nonlinear combination
        indices_seed = proof_stream.prover_fiat_shamir()
        indices = BrainfuckStark.sample_indices(
            self.security_level, indices_seed, self.fri.domain.length)

        unit_distances = [table.unit_distance(
            self.fri.domain.length) for table in self.tables]
        unit_distances = list(set(unit_distances))

        # open leafs of zipped codewords at indicated positions
        for index in indices:
            for distance in [0] + unit_distances:
                idx = (index + distance) % self.fri.domain.length
                element = base_tree.leafs[idx][0]
                salt, path = base_tree.open(idx)
                proof_stream.push(element)
                proof_stream.push((salt, path))

                assert (SaltedMerkle.verify(base_tree.root(), idx, salt, path,
                                            element)), "SaltedMerkle for base tree leaf fails to verify"

                proof_stream.push(extension_tree.leafs[idx][0])
                proof_stream.push(extension_tree.open(idx))

        # open combination codeword at the same positions
        for index in indices:
            proof_stream.push(combination_tree.leafs[index])
            proof_stream.push(combination_tree.open(index))
            assert (Merkle.verify(combination_tree.root(), index,
                                  combination_tree.open(index), combination_tree.leafs[index]))

        # prove low degree of combination polynomial, and collect indices
        indices = self.fri.prove(combination_codeword, proof_stream)

        # the final proof is just the serialized stream
        ret = proof_stream.serialize()

        return ret

    def verify(self, proof, proof_stream=None):

        verifier_verdict = True

        # deserialize with right proof stream
        if proof_stream == None:
            proof_stream = ProofStream()
        proof_stream = proof_stream.deserialize(proof)

        # get Merkle root of base tables
        base_root = proof_stream.pull()

        # get coefficients for table extensions
        challenges = self.sample_weights(
            11, proof_stream.verifier_fiat_shamir())

        # get root of table extensions
        extension_root = proof_stream.pull()

        # get terminals
        # TODO: drop names; just get four terminals from proof stream
        processor_instruction_permutation_terminal = proof_stream.pull()
        processor_memory_permutation_terminal = proof_stream.pull()
        processor_input_evaluation_terminal = proof_stream.pull()
        processor_output_evaluation_terminal = proof_stream.pull()
        instruction_evaluation_terminal = proof_stream.pull()
        terminals = [processor_instruction_permutation_terminal,
                     processor_memory_permutation_terminal,
                     processor_input_evaluation_terminal,
                     processor_output_evaluation_terminal,
                     instruction_evaluation_terminal]

        base_degree_bounds = reduce(lambda x, y: x + y,
                                    [[table.interpolant_degree(
                                    )] * table.base_width for table in self.tables],
                                    [])

        extension_degree_bounds = reduce(lambda x, y: x+y,
                                         [[table.interpolant_degree()] * (table.full_width - table.base_width)
                                          for table in self.tables],
                                         [])

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every other polynomial (base, extension, quotients)
        num_base_polynomials = sum(
            table.base_width for table in self.tables)
        num_extension_polynomials = sum(
            table.full_width - table.base_width for table in self.tables)
        num_randomizer_polynomials = 1

        num_quotient_polynomials = sum(table.num_quotients(
            challenges, terminals) for table in self.tables)

        num_difference_quotients = len(self.permutation_arguments)

        weights_seed = proof_stream.verifier_fiat_shamir()
        weights = self.sample_weights(
            num_randomizer_polynomials +
            2*num_base_polynomials +
            2*num_extension_polynomials +
            2*num_quotient_polynomials +
            2*num_difference_quotients,
            weights_seed)

        # pull Merkle root of combination codeword
        combination_root = proof_stream.pull()

        # get indices of leafs to verify nonlinear combinatoin
        indices_seed = proof_stream.verifier_fiat_shamir()
        indices = BrainfuckStark.sample_indices(
            self.security_level, indices_seed, self.fri.domain.length)

        unit_distances = [table.unit_distance(
            self.fri.domain.length) for table in self.tables]
        unit_distances = list(set(unit_distances))

        # get leafs at indicated positions
        tuples = dict()
        for index in indices:
            for distance in [0] + unit_distances:
                idx = (index + distance) % self.fri.domain.length

                element = proof_stream.pull()
                salt, path = proof_stream.pull()
                verifier_verdict = verifier_verdict and SaltedMerkle.verify(
                    base_root, idx, salt, path, element)
                tuples[idx] = [self.xfield.lift(e) for e in list(element)]
                assert (
                    verifier_verdict), "salted base tree verify must succeed for base codewords"

                element = proof_stream.pull()
                salt, path = proof_stream.pull()
                verifier_verdict = verifier_verdict and SaltedMerkle.verify(
                    extension_root, idx, salt, path, element)
                tuples[idx] = tuples[idx] + list(element)
                assert (
                    verifier_verdict), "salted base tree verify must succeed for extension codewords"

        assert (num_base_polynomials == len(base_degree_bounds)
                ), f"number of base polynomials {num_base_polynomials} =/= number of base degree bounds {len(base_degree_bounds)}"
        # verify nonlinear combination
        for index in indices:
            # collect terms: randomizer
            terms: list[ExtensionFieldElement] = tuples[index][0:num_randomizer_polynomials]

            # collect terms: base
            for i in range(num_randomizer_polynomials, num_randomizer_polynomials+num_base_polynomials):
                terms += [tuples[index][i]]
                shift = self.max_degree - \
                    base_degree_bounds[i-num_randomizer_polynomials]
                terms += [tuples[index][i] *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]

            # collect terms: extension
            extension_offset = num_randomizer_polynomials + \
                sum(table.base_width for table in self.tables)

            assert (len(
                terms) == 2 * extension_offset - num_randomizer_polynomials), f"number of terms {len(terms)} does not match with extension offset {2 * extension_offset - num_randomizer_polynomials}"

            for i in range(num_extension_polynomials):
                terms += [tuples[index][extension_offset+i]]
                shift = self.max_degree - extension_degree_bounds[i]
                terms += [tuples[index][extension_offset+i]
                          * self.xfield.lift(self.fri.domain(index) ^ shift)]

            # collect terms: quotients
            # quotients need to be computed
            acc_index = num_randomizer_polynomials
            points = []
            for table in self.tables:
                step = table.base_width
                points += [tuples[index][acc_index:(acc_index+step)]]
                acc_index += step

            assert (acc_index == extension_offset,
                    "Column count in verifier must match until extension columns")

            for point, table in zip(points, self.tables):
                step = table.full_width - table.base_width
                point += tuples[index][acc_index:(acc_index+step)]
                acc_index += step

            assert (acc_index == len(
                tuples[index]), "Column count in verifier must match until end")

            base_acc_index = num_randomizer_polynomials
            ext_acc_index = extension_offset
            for point, table in zip(points, self.tables):
                # boundary
                for constraint, bound in zip(table.boundary_constraints_ext(challenges), table.boundary_quotient_degree_bounds(challenges)):
                    eval = constraint.evaluate(point)
                    quotient = eval / \
                        (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
                    terms += [quotient]
                    shift = self.max_degree - bound
                    terms += [quotient *
                              self.xfield.lift(self.fri.domain(index) ^ shift)]

                # transition
                unit_distance = table.unit_distance(
                    self.fri.domain.length)
                next_index = (index + unit_distance) % self.fri.domain.length
                next_point = tuples[next_index][base_acc_index:(
                    base_acc_index+table.base_width)]
                next_point += tuples[next_index][ext_acc_index:(
                    ext_acc_index+table.full_width-table.base_width)]
                base_acc_index += table.base_width
                ext_acc_index += table.full_width - table.base_width
                for constraint, bound in zip(table.transition_constraints_ext(challenges), table.transition_quotient_degree_bounds(challenges)):
                    eval = constraint.evaluate(
                        point + next_point)
                    # If height == 0, then there is no subgroup where the transition polynomials should be zero.
                    # The fast zerofier (based on group theory) needs a non-empty group.
                    # Forcing it on an empty group generates a division by zero error.
                    if table.height == 0:
                        quotient = self.xfield.zero()
                    else:
                        quotient = eval * self.xfield.lift(self.fri.domain(index) - table.omicron.inverse()) / (
                            self.xfield.lift(self.fri.domain(index) ^ table.height) - self.xfield.one())
                    terms += [quotient]
                    shift = self.max_degree - bound
                    terms += [quotient *
                              self.xfield.lift(self.fri.domain(index) ^ shift)]

                # terminal
                for constraint, bound in zip(table.terminal_constraints_ext(challenges, terminals), table.terminal_quotient_degree_bounds(challenges, terminals)):
                    eval = constraint.evaluate(point)
                    quotient = eval / \
                        (self.xfield.lift(self.fri.domain(index)) -
                         self.xfield.lift(table.omicron.inverse()))
                    terms += [quotient]
                    shift = self.max_degree - bound
                    terms += [quotient *
                              self.xfield.lift(self.fri.domain(index) ^ shift)]

            for arg in self.permutation_arguments:
                quotient = arg.evaluate_difference(
                    points) / (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                degree_bound = arg.quotient_degree_bound()
                shift = self.max_degree - degree_bound
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]

            assert (len(terms) == len(
                weights)), f"length of terms ({len(terms)}) must be equal to length of weights ({len(weights)})"

            # compute inner product of weights and terms
            inner_product = reduce(
                lambda x, y: x + y, [w * t for w, t in zip(weights, terms)], self.xfield.zero())

            # get value of the combination codeword to test the inner product against
            combination_leaf = proof_stream.pull()
            combination_path = proof_stream.pull()

            # verify Merkle authentication path
            verifier_verdict = verifier_verdict and Merkle.verify(
                combination_root, index, combination_path, combination_leaf)
            if not verifier_verdict:
                return False

            # check equality
            verifier_verdict = verifier_verdict and combination_leaf == inner_product
            if not verifier_verdict:
                return False

        # verify low degree of combination polynomial
        verifier_verdict = self.fri.verify(proof_stream, combination_root)

        # verify external terminals:
        for ea in self.evaluation_arguments:
            verifier_verdict = verifier_verdict and ea.select_terminal(
                terminals) == ea.compute_terminal(challenges)

        return verifier_verdict
