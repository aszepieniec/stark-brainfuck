from binascii import hexlify
from extension_field import ExtensionField, ExtensionFieldElement
from ip import ProofStream
from merkle import Merkle
from fri import *
from instruction_table import InstructionTable
from io_table import InputTable, OutputTable
from labeled_list import LabeledList
from memory_table import MemoryTable
from permutation_argument import PermutationArgument
from processor_table import ProcessorTable
from salted_merkle import SaltedMerkle
from univariate import *
from multivariate import *
from ntt import *
from functools import reduce
import os
import time

from vm import VirtualMachine


class BrainfuckStark:
    field = BaseField.main()
    xfield = ExtensionField.main()

    def __init__(self, running_time, program, input_symbols, output_symbols):
        # set fields of computational integrity claim
        self.running_time = running_time
        self.program = program
        self.input_symbols = input_symbols
        self.output_symbols = output_symbols

        # set parameters
        self.expansion_factor = 16  # for security and compactness
        self.expansion_factor = 4  # for speed
        self.num_colinearity_checks = 40  # for security
        self.num_colinearity_checks = 1  # for speed
        self.security_level = 160  # for security
        self.security_level = 1  # for speed
        assert(self.expansion_factor & (self.expansion_factor - 1)
               == 0), "expansion factor must be a power of 2"
        assert(self.expansion_factor >=
               4), "expansion factor must be 4 or greater"
        assert(self.num_colinearity_checks * len(bin(self.expansion_factor)
               [3:]) >= self.security_level), "number of colinearity checks times log of expansion factor must be at least security level"

        self.num_randomizers = 1  # TODO: self.security_level

        self.vm = VirtualMachine()

        # instantiate table objects
        order = 1 << 32
        smooth_generator = BrainfuckStark.field.primitive_nth_root(order)

        self.processor_table = ProcessorTable(
            self.field, running_time, self.num_randomizers, smooth_generator, order)
        self.instruction_table = InstructionTable(
            self.field, running_time + len(program), self.num_randomizers, smooth_generator, order)
        self.memory_table = MemoryTable(
            self.field, running_time, self.num_randomizers, smooth_generator, order)
        self.input_table = InputTable(
            self.field, len(input_symbols), smooth_generator, order)
        self.output_table = OutputTable(
            self.field, len(output_symbols), smooth_generator, order)

        self.base_tables = [self.processor_table, self.instruction_table,
                            self.memory_table, self.input_table, self.output_table]

        # instantiate permutation objects
        processor_instruction_permutation = PermutationArgument(
            self.base_tables, (0, ProcessorTable.instruction_permutation), (1, InstructionTable.permutation))
        processor_memory_permutation = PermutationArgument(
            self.base_tables, (0, ProcessorTable.memory_permutation), (2, MemoryTable.permutation))
        self.permutation_arguments = [
            processor_instruction_permutation, processor_memory_permutation]

        # compute self.fri domain length
        self.max_degree = 1
        for table in self.base_tables:
            for air in table.base_transition_constraints():
                degree_bounds = [table.interpolant_degree()] * \
                    table.base_width * 2
                degree = air.symbolic_degree_bound(
                    degree_bounds) - (table.height - 1)
                if self.max_degree < degree:
                    self.max_degree = degree

        self.max_degree = BrainfuckStark.roundup_npo2(self.max_degree) - 1
        fri_domain_length = (self.max_degree+1) * self.expansion_factor

        print("max degree:", self.max_degree)
        print("fri domain length:", fri_domain_length)

        # instantiate self.fri object
        generator = BrainfuckStark.field.generator()
        omega = BrainfuckStark.field.primitive_nth_root(fri_domain_length)
        self.fri = Fri(generator, omega, fri_domain_length,
                       self.expansion_factor, self.num_colinearity_checks, self.xfield)

    def get_terminals(self):
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

    def prove(self, running_time, program, processor_matrix, instruction_matrix, input_matrix, output_matrix, proof_stream=None):
        assert(running_time == len(processor_matrix))
        assert(running_time + len(program) == len(instruction_matrix))

        start_time_prover = time.time()

        # populate tables' matrices
        self.processor_table.matrix = processor_matrix
        self.instruction_table.matrix = instruction_matrix
        self.input_table.matrix = input_matrix
        self.output_table.matrix = output_matrix

        # pad table to height 2^k
        self.processor_table.pad()
        self.instruction_table.pad()
        self.input_table.pad()
        self.output_table.pad()

        # instantiate other table objects
        self.memory_table.matrix = MemoryTable.derive_matrix(
            self.processor_table.matrix, self.num_randomizers)

        # create proof stream if we don't have it already
        if proof_stream == None:
            proof_stream = ProofStream()

        # print("interpolating base tables ...")
        # tick = time.time()

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
            lambda x, y: x+y, [table.lde(self.fri.domain) for table in self.base_tables], [])
        all_base_codewords = randomizer_codewords + base_codewords

        base_degree_bounds = reduce(
            lambda x, y: x+y, [[table.interpolant_degree()] * table.base_width for table in self.base_tables], [])

        zipped_codeword = list(zip(*all_base_codewords))
        base_tree = SaltedMerkle(zipped_codeword)
        proof_stream.push(base_tree.root())
        # tock = time.time()
        # print("commitment to base polynomials took", (tock - tick), "seconds")

        # get coefficients for table extensions
        challenges = self.sample_weights(
            11, proof_stream.prover_fiat_shamir())

        initials = [self.xfield.sample(os.urandom(3*8))
                    for i in range(len(self.permutation_arguments))]

        for table in self.base_tables:
            table.extend(challenges, initials)

        terminals = self.get_terminals()

        extension_codewords = reduce(
            lambda x, y: x+y, [table.ldex(self.fri.domain, self.xfield) for table in self.base_tables], [])

        zipped_extension_codeword = list(zip(*extension_codewords))
        extension_tree = SaltedMerkle(zipped_extension_codeword)
        proof_stream.push(extension_tree.root())
        print("-> extension tree root:", hexlify(extension_tree.root()))
        # tock = time.time()

        extension_degree_bounds = reduce(lambda x, y: x+y, [[table.interpolant_degree()] * (
            table.full_width - table.base_width) for table in self.base_tables], [])

        quotient_codewords = LabeledList()

        for table in self.base_tables:
            quotient_codewords.concatenate(table.all_quotients_labeled(
                self.fri.domain, table.codewords, challenges, terminals))

        quotient_degree_bounds = LabeledList()
        for table in self.base_tables:
            quotient_degree_bounds.concatenate(
                table.all_quotient_degree_bounds_labeled(challenges, terminals))

        # ... and equal initial values
        for pa in self.permutation_arguments:
            quotient_codewords.append(pa.quotient(self.fri.domain), "diff")
            quotient_degree_bounds.append(
                pa.quotient_degree_bound(), "diff")

        for t in terminals:
            proof_stream.push(t)

        # get weights for nonlinear combination
        #  - 1 for randomizer polynomials
        #  - 2 for every other polynomial (base, extension, quotients)
        num_base_polynomials = sum(
            table.base_width for table in self.base_tables)
        num_extension_polynomials = sum(
            table.full_width - table.base_width for table in self.base_tables)
        num_randomizer_polynomials = 1
        num_quotient_polynomials = len(quotient_degree_bounds)
        weights_seed = proof_stream.prover_fiat_shamir()
        weights = self.sample_weights(
            num_randomizer_polynomials
            + 2 * (num_base_polynomials +
                   num_extension_polynomials +
                   num_quotient_polynomials),
            weights_seed)

        print("** challenges for weights")

        # compute terms of nonlinear combination polynomial
        # TODO: memoize shifted self.fri domains
        # print("computing nonlinear combination ...")
        # tick = time.time()
        terms = [randomizer_codeword]
        # base_codewords = processor_base_codewords + instruction_base_codewords + \
        # memory_base_codewords + input_base_codewords + output_base_codewords
        assert(len(base_codewords) ==
               num_base_polynomials), f"number of base codewords {len(base_codewords)} codewords =/= number of base polynomials {num_base_polynomials}!"
        for i in range(len(base_codewords)):
            terms += [[self.xfield.lift(c) for c in base_codewords[i]]]
            shift = self.max_degree - base_degree_bounds[i]
            print("prover shift for base codeword", i, ":", shift)
            # print("max degree:", max_degree)
            # print("bound:", base_degree_bounds[i])
            # print("term:", terms[-1][0])
            terms += [[self.xfield.lift((self.fri.domain(j) ^ shift) * base_codewords[i][j])
                      for j in range(self.fri.domain.length)]]
            if os.environ.get('DEBUG') is not None:
                print(f"before domain interpolation")
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                print(
                    f"degree of interpolation, base_codewords({i}): {interpolated.degree()}")
                assert(interpolated.degree() <= self.max_degree)
        assert(len(extension_codewords) ==
               num_extension_polynomials), f"number of extension codewords {len(extension_codewords)} =/= number of extension polynomials {num_extension_polynomials}"
        for i in range(len(extension_codewords)):
            terms += [extension_codewords[i]]
            shift = self.max_degree - extension_degree_bounds[i]
            print("prover shift for extension codeword", i, ": ", shift)
            terms += [[self.xfield.lift(self.fri.domain(j) ^ shift) * extension_codewords[i][j]
                      for j in range(self.fri.domain.length)]]
            if os.environ.get('DEBUG') is not None:
                print(f"before domain interpolation")
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                print(
                    f"degree of interpolation, extension_codewords({i}): {interpolated.degree()}")
                assert(interpolated.degree() <= self.max_degree)
        assert(len(quotient_codewords) ==
               num_quotient_polynomials), f"number of quotient codewords {len(quotient_codewords)} =/= number of quotient polynomials {num_quotient_polynomials}"

        for i in range(len(quotient_codewords)):
            quotient_codeword_i = quotient_codewords.get(i)
            quotient_degree_bound_i = quotient_degree_bounds.get(i)
            terms += [quotient_codeword_i]
            print("", i, ".", quotient_codewords.label(i))
            print("", i, ".", quotient_degree_bounds.label(i))
            print("quotient_codewords codeword shift: ", shift)
            print("quotient_degree_bounds : ", quotient_degree_bound_i)
            print()
            if os.environ.get('DEBUG') is not None:
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                assert(interpolated.degree() == -1 or interpolated.degree() <=
                       quotient_degree_bound_i), f"for unshifted quotient polynomial {i}, interpolated degree is {interpolated.degree()} but > degree bound i = {quotient_degree_bound_i}"
            shift = self.max_degree - quotient_degree_bound_i
            print("prover shift for quotient", i, ":", shift)

            terms += [[self.xfield.lift(self.fri.domain(j) ^ shift) * quotient_codeword_i[j]
                      for j in range(self.fri.domain.length)]]
            if os.environ.get('DEBUG') is not None:
                print(f"before domain interpolation")
                interpolated = self.fri.domain.xinterpolate(terms[-1])
                print(
                    f"degree of interpolation, , quotient_codewords({i}): {interpolated.degree()}")
                print("quotient  degree bound:", quotient_degree_bound_i)
                assert(interpolated.degree(
                ) == -1 or interpolated.degree() <= self.max_degree), f"for (shifted) quotient polynomial {i}, interpolated degree is {interpolated.degree()} but > max_degree = {self.max_degree}"
        # print("got terms after", (time.time() - tick), "seconds")

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for i)
        assert(len(terms) == len(
            weights)), f"number of terms {len(terms)} is not equal to number of weights {len(weights)}"

        print("number of terms:", len(terms))
        print("number of weights:", len(weights))

        combination_codeword = reduce(
            lambda lhs, rhs: [l+r for l, r in zip(lhs, rhs)], [[w * e for e in t] for w, t in zip(weights, terms)], [self.xfield.zero()] * self.fri.domain.length)
        # print("finished computing nonlinear combination; calculation took", time.time() - tick, "seconds")

        # print("prover terms:")
        # for t in terms:
        #     print(t[0])
        # print("prover weights:")
        # for w in weights:
        #     print(w)

        # commit to combination codeword
        combination_tree = Merkle(combination_codeword)
        proof_stream.push(combination_tree.root())
        print("-> combination codeword tree root")

        # get indices of leafs to prove nonlinear combination
        indices_seed = proof_stream.prover_fiat_shamir()
        print("prover indices seed:", hexlify(indices_seed))
        print("** indices for nonlicombo")
        indices = BrainfuckStark.sample_indices(
            self.security_level, indices_seed, self.fri.domain.length)

        # indices = [0]  # TODO remove me when not debugging

        unit_distances = [table.unit_distance(
            self.fri.domain.length) for table in self.base_tables]
        unit_distances = list(set(unit_distances))

        # open leafs of zipped codewords at indicated positions
        for index in indices:
            print("I think the index is", index)
            for distance in [0] + unit_distances:
                idx = (index + distance) % self.fri.domain.length
                print("I think idx is", idx)
                element = base_tree.leafs[idx][0]
                salt, path = base_tree.open(idx)
                proof_stream.push(element)
                proof_stream.push((salt, path))
                print("-> base leafs and path for index", index, "+",
                      distance, "=", idx, "mod", self.fri.domain.length)

                assert(SaltedMerkle.verify(base_tree.root(), idx, salt, path,
                       element)), "SaltedMerkle for base tree leaf fails to verify"

                proof_stream.push(extension_tree.leafs[idx][0])
                proof_stream.push(extension_tree.open(idx))
                print("-> extension leafs and path for index", index, "+",
                      distance, "=", idx, "mod", self.fri.domain.length, "\n")

        # open combination codeword at the same positions
        for index in indices:
            if index == indices[0]:
                print("-> combination root:", hexlify(combination_tree.root()))
                print("-> combination path:", [hexlify(p)
                      for p in combination_tree.open(index)])
                print("-> combination index:", index)
                print("-> combination leaf:", combination_tree.leafs[index])
            print("prover proof length: ", len(proof_stream.objects))
            proof_stream.push(combination_tree.leafs[index])
            print("prover proof length: ", len(proof_stream.objects))
            proof_stream.push(combination_tree.open(index))
            assert(Merkle.verify(combination_tree.root(), index,
                   combination_tree.open(index), combination_tree.leafs[index]))
            print("prover proof length: ", len(proof_stream.objects))
            print("printed path: ", [hexlify(p)
                  for p in combination_tree.open(index)])

        # prove low degree of combination polynomial, and collect indices
        tick = time.time()
        print("starting self.fri")
        indices = self.fri.prove(combination_codeword, proof_stream)
        tock = time.time()
        print("FRI took ", (tock - tick), "seconds")

        # the final proof is just the serialized stream
        ret = proof_stream.serialize()
        end_time_prover = time.time()
        print("STARK proof took ", (end_time_prover - start_time_prover), "seconds")

        return ret

    def verify(self, proof, proof_stream=None):
        print("inside verifier \\o/")

        verifier_verdict = True

        # deserialize with right proof stream
        if proof_stream == None:
            proof_stream = ProofStream()
        proof_stream = proof_stream.deserialize(proof)

        # get Merkle root of base tables
        base_root = proof_stream.pull()
        print("<- base tree root:", hexlify(base_root))

        # get coefficients for table extensions
        challenges = self.sample_weights(
            11, proof_stream.verifier_fiat_shamir())
        print("** challenges for extension coefficients")

        # get root of table extensions
        extension_root = proof_stream.pull()

        # get terminals
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
                                    )] * table.base_width for table in self.base_tables],
                                    [])

        extension_degree_bounds = reduce(lambda x, y: x+y,
                                         [[table.interpolant_degree()] * (table.full_width - table.base_width)
                                          for table in self.base_tables],
                                         [])

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every other polynomial (base, extension, quotients)
        num_base_polynomials = sum(
            table.base_width for table in self.base_tables)
        num_extension_polynomials = sum(
            table.full_width - table.base_width for table in self.base_tables)
        num_randomizer_polynomials = 1

        # num_quotient_polynomials = processor_extension.num_quotients()
        # num_quotient_polynomials += instruction_extension.num_quotients()
        # num_quotient_polynomials += memory_extension.num_quotients()
        # num_quotient_polynomials += input_extension.num_quotients()
        # num_quotient_polynomials += output_extension.num_quotients()
        num_quotient_polynomials = sum(table.num_quotients(
            challenges, terminals) for table in self.base_tables)

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
        print("<- combination codeword root")

        # get indices of leafs to verify nonlinear combinatoin
        indices_seed = proof_stream.verifier_fiat_shamir()
        print("verifier's indices seed:", hexlify(indices_seed))
        print("** indices for nonlicombo")
        indices = BrainfuckStark.sample_indices(
            self.security_level, indices_seed, self.fri.domain.length)

        # indices = [0]  # TODO remove me when not debugging

        unit_distances = [table.unit_distance(
            self.fri.domain.length) for table in self.base_tables]
        unit_distances = list(set(unit_distances))

        # get leafs at indicated positions
        tuples = dict()
        for index in indices:
            for distance in [0] + unit_distances:
                idx = (index + distance) % self.fri.domain.length
                print("idx:", idx)

                element = proof_stream.pull()
                salt, path = proof_stream.pull()
                verifier_verdict = verifier_verdict and SaltedMerkle.verify(
                    base_root, idx, salt, path, element)
                tuples[idx] = [self.xfield.lift(e) for e in list(element)]
                assert(
                    verifier_verdict), "salted base tree verify must succeed for base codewords"

                element = proof_stream.pull()
                salt, path = proof_stream.pull()
                verifier_verdict = verifier_verdict and SaltedMerkle.verify(
                    extension_root, idx, salt, path, element)
                tuples[idx] = tuples[idx] + list(element)
                assert(
                    verifier_verdict), "salted base tree verify must succeed for extension codewords"

                print("<- leafs and path for index", index, "+",
                      distance, "=", idx, "mod", self.fri.domain.length)

        assert(num_base_polynomials == len(base_degree_bounds)
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
            print("len(terms) after base codewords: ", len(terms))

            # collect terms: extension
            extension_offset = num_randomizer_polynomials + \
                sum(table.base_width for table in self.base_tables)

            assert(len(
                terms) == 2 * extension_offset - num_randomizer_polynomials), f"number of terms {len(terms)} does not match with extension offset {2 * extension_offset - num_randomizer_polynomials}"

            for i in range(num_extension_polynomials):
                print("keys of tuples:", [k for k in tuples.keys()])
                print("extension offset plus i = ", extension_offset + i)
                print("i:", i)
                print("index:", index)
                print("len of tuples at index:", len(tuples[index]))
                terms += [tuples[index][extension_offset+i]]
                shift = self.max_degree - extension_degree_bounds[i]
                print("extension degree shift: ", shift)
                terms += [tuples[index][extension_offset+i]
                          * self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after extension codewords: ", len(terms))

            # collect terms: quotients
            # quotients need to be computed

            acc_index = num_randomizer_polynomials
            processor_point = tuples[index][acc_index:(
                acc_index+self.processor_table.base_width)]
            acc_index += self.processor_table.base_width
            instruction_point = tuples[index][acc_index:(
                acc_index+self.instruction_table.base_width)]
            acc_index += self.instruction_table.base_width
            memory_point = tuples[index][(acc_index):(
                acc_index+self.memory_table.base_width)]
            acc_index += self.memory_table.base_width
            input_point = tuples[index][(acc_index):(
                acc_index+self.input_table.base_width)]
            acc_index += self.input_table.base_width
            output_point = tuples[index][(acc_index):(
                acc_index+self.output_table.base_width)]
            acc_index += self.output_table.base_width

            assert(acc_index == extension_offset,
                   "Column count in verifier must match until extension columns")

            acc_index = extension_offset  # Should be unchanged!!
            processor_point += tuples[index][acc_index:(
                acc_index+self.processor_table.full_width-self.processor_table.base_width)]
            acc_index += self.processor_table.full_width-self.processor_table.base_width

            instruction_point += tuples[index][(acc_index):(
                acc_index+self.instruction_table.full_width-self.instruction_table.base_width)]
            acc_index += self.instruction_table.full_width-self.instruction_table.base_width

            memory_point += tuples[index][(acc_index):(
                acc_index+self.memory_table.full_width-self.memory_table.base_width)]
            acc_index += self.memory_table.full_width-self.memory_table.base_width

            input_point += tuples[index][(acc_index):(
                acc_index+self.input_table.full_width-self.input_table.base_width)]
            acc_index += self.input_table.full_width-self.input_table.base_width

            output_point += tuples[index][(acc_index):(
                acc_index+self.output_table.full_width-self.output_table.base_width)]
            acc_index += self.output_table.full_width-self.output_table.base_width
            assert(acc_index == len(
                tuples[index]), "Column count in verifier must match until end")

            # ******************** processor quotients ********************
            # boundary
            print("type of points:", ",".join(str(type(p))
                  for p in processor_point))
            for constraint, bound in zip(self.processor_table.boundary_constraints_ext(challenges), self.processor_table.boundary_quotient_degree_bounds(challenges)):
                print("type of constraint:", type(constraint), "over",
                      type(list(constraint.dictionary.values())[0]))
                eval = constraint.evaluate(processor_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after processor boundaries: ", len(terms))

            # transition
            unit_distance = self.processor_table.unit_distance(
                self.fri.domain.length)
            next_index = (index + unit_distance) % self.fri.domain.length
            next_processor_point = tuples[next_index][num_randomizer_polynomials:(
                num_randomizer_polynomials+self.processor_table.base_width)]
            next_processor_point += tuples[next_index][extension_offset:(
                extension_offset+self.processor_table.full_width-self.processor_table.base_width)]
            for constraint, bound in zip(self.processor_table.transition_constraints_ext(challenges), self.processor_table.transition_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(
                    processor_point + next_processor_point)
                quotient = eval * self.xfield.lift(self.fri.domain(index) - self.processor_table.omicron.inverse()) / (
                    self.xfield.lift(self.fri.domain(index) ^ self.processor_table.height) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after processor transitions: ", len(terms))

            # terminal
            for constraint, bound in zip(self.processor_table.terminal_constraints_ext(challenges, terminals), self.processor_table.terminal_quotient_degree_bounds(challenges, terminals)):
                eval = constraint.evaluate(processor_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) -
                     self.xfield.lift(self.processor_table.omicron.inverse()))
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after processor terminals: ", len(terms))

            # ******************** instruction quotients ********************
            # boundary
            for constraint, bound in zip(self.instruction_table.boundary_constraints_ext(challenges), self.instruction_table.boundary_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(instruction_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after instruction boundaries: ", len(terms))

            # transition
            unit_distance = self.instruction_table.unit_distance(
                self.fri.domain.length)
            next_index = (index + unit_distance) % self.fri.domain.length
            next_instruction_point = tuples[next_index][(num_randomizer_polynomials+self.processor_table.base_width):(
                num_randomizer_polynomials+self.processor_table.base_width+self.instruction_table.base_width)]
            next_instruction_point += tuples[next_index][(extension_offset+self.processor_table.full_width-self.processor_table.base_width):(
                extension_offset+self.processor_table.full_width-self.processor_table.base_width+self.instruction_table.full_width-self.instruction_table.base_width)]
            for constraint, bound in zip(self.instruction_table.transition_constraints_ext(challenges), self.instruction_table.transition_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(
                    instruction_point + next_instruction_point)
                quotient = eval * self.xfield.lift(self.fri.domain(index) - self.instruction_table.omicron.inverse()) / (
                    self.xfield.lift(self.fri.domain(index) ^ BrainfuckStark.roundup_npo2(self.instruction_table.height)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after instruction transitions: ", len(terms))

            # terminal
            for constraint, bound in zip(self.instruction_table.terminal_constraints_ext(challenges, terminals), self.instruction_table.terminal_quotient_degree_bounds(challenges, terminals)):
                eval = constraint.evaluate(instruction_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) -
                     self.xfield.lift(self.instruction_table.omicron.inverse()))
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after instruction terminals: ", len(terms))

            # ******************** memory quotients ********************
            # boundary
            for constraint, bound in zip(self.memory_table.boundary_constraints_ext(challenges), self.memory_table.boundary_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(memory_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]

            # transition
            unit_distance = self.memory_table.unit_distance(
                self.fri.domain.length)
            next_index = (index + unit_distance) % self.fri.domain.length
            next_memory_point = tuples[next_index][(num_randomizer_polynomials+self.processor_table.base_width+self.instruction_table.base_width):(
                num_randomizer_polynomials+self.processor_table.base_width+self.instruction_table.base_width+self.memory_table.base_width)]
            next_memory_point += tuples[next_index][(extension_offset+self.processor_table.full_width-self.processor_table.base_width+self.instruction_table.full_width-self.instruction_table.base_width):(
                extension_offset+self.processor_table.full_width-self.processor_table.base_width+self.instruction_table.full_width-self.instruction_table.base_width+self.memory_table.full_width-self.memory_table.base_width)]
            for constraint, bound in zip(self.memory_table.transition_constraints_ext(challenges), self.memory_table.transition_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(memory_point + next_memory_point)
                quotient = eval * self.xfield.lift(self.fri.domain(index) - self.memory_table.omicron.inverse()) / (
                    self.xfield.lift(self.fri.domain(index) ^ BrainfuckStark.roundup_npo2(self.memory_table.height)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]

            # terminal
            for constraint, bound in zip(self.memory_table.terminal_constraints_ext(challenges, terminals), self.memory_table.terminal_quotient_degree_bounds(challenges, terminals)):
                eval = constraint.evaluate(memory_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) -
                     self.xfield.lift(self.memory_table.omicron.inverse()))
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after memory: ", len(terms))

            # ******************** input quotients ********************
            # boundary
            for constraint, bound in zip(self.input_table.boundary_constraints_ext(challenges), self.input_table.boundary_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(input_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after input boundaries: ", len(terms))

            # transition
            unit_distance = self.input_table.unit_distance(
                self.fri.domain.length)
            next_index = (index + unit_distance) % self.fri.domain.length
            next_input_point = tuples[next_index][(num_randomizer_polynomials+self.processor_table.base_width+self.instruction_table.base_width+self.memory_table.base_width):(
                num_randomizer_polynomials+self.processor_table.base_width+self.instruction_table.base_width+self.memory_table.base_width+self.input_table.base_width)]
            next_input_point += tuples[next_index][(extension_offset+self.processor_table.full_width-self.processor_table.base_width+self.instruction_table.full_width-self.instruction_table.base_width+self.memory_table.full_width-self.memory_table.base_width):(
                extension_offset+self.processor_table.full_width-self.processor_table.base_width+self.instruction_table.full_width-self.instruction_table.base_width+self.memory_table.full_width-self.memory_table.base_width+self.input_table.full_width-self.input_table.base_width)]
            print("Hi from input transition quotient loop")
            for constraint, bound in zip(self.input_table.transition_constraints_ext(challenges), self.input_table.transition_quotient_degree_bounds(challenges)):
                print("Hi from input transition quotient loop")
                eval = constraint.evaluate(input_point + next_input_point)
                quotient = eval * self.xfield.lift(self.fri.domain(index) - self.input_table.omicron.inverse()) / (
                    self.xfield.lift(self.fri.domain(index) ^ BrainfuckStark.roundup_npo2(self.input_table.height)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after input transitions: ", len(terms))

            # terminal
            for constraint, bound in zip(self.input_table.terminal_constraints_ext(challenges, terminals), self.input_table.terminal_quotient_degree_bounds(challenges, terminals)):
                eval = constraint.evaluate(input_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) -
                     self.xfield.lift(self.input_table.omicron.inverse()))
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after input terminals: ", len(terms))

            # ******************** output quotients ********************
            # boundary
            for constraint, bound in zip(self.output_table.boundary_constraints_ext(challenges), self.output_table.boundary_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(output_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after output boundaries: ", len(terms))

            # transition
            unit_distance = self.output_table.unit_distance(
                self.fri.domain.length)
            next_index = (index + unit_distance) % self.fri.domain.length
            base_start_index = self.processor_table.base_width + \
                self.instruction_table.base_width + self.memory_table.base_width
            base_start_index += self.output_table.base_width
            next_output_point = tuples[next_index][(num_randomizer_polynomials+base_start_index):(
                num_randomizer_polynomials+base_start_index+self.output_table.base_width)]
            extension_start_index = self.processor_table.full_width - self.processor_table.base_width + \
                self.instruction_table.full_width - self.instruction_table.base_width + \
                self.memory_table.full_width - self.memory_table.base_width
            extension_start_index += self.output_table.full_width - self.output_table.base_width
            next_output_point += tuples[next_index][(extension_offset+extension_start_index):(
                extension_offset+extension_start_index+self.output_table.full_width-self.output_table.base_width)]
            for constraint, bound in zip(self.output_table.transition_constraints_ext(challenges), self.output_table.transition_quotient_degree_bounds(challenges)):
                eval = constraint.evaluate(
                    output_point + next_output_point)
                quotient = eval * self.xfield.lift(self.fri.domain(index) - self.output_table.omicron.inverse()) / (
                    self.xfield.lift(self.fri.domain(index) ^ BrainfuckStark.roundup_npo2(self.output_table.height)) - self.xfield.one())
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after output transitions: ", len(terms))

            # terminal
            for constraint, bound in zip(self.output_table.terminal_constraints_ext(challenges, terminals), self.output_table.terminal_quotient_degree_bounds(challenges, terminals)):
                eval = constraint.evaluate(output_point)
                quotient = eval / \
                    (self.xfield.lift(self.fri.domain(index)) -
                     self.xfield.lift(self.output_table.omicron.inverse()))
                terms += [quotient]
                shift = self.max_degree - bound
                print("verifier shift:", shift)
                terms += [quotient *
                          self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after output terminals: ", len(terms))

            # ******************** difference quotients ********************
            difference = (processor_point[ProcessorTable.instruction_permutation] -
                          instruction_point[InstructionTable.permutation])
            quotient = difference / \
                (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
            terms += [quotient]
            # shift = self.max_degree - \
            #     (BrainfuckStark.roundup_npo2(
            #         self.running_time + len(self.program)) + self.num_randomizers - 2)
            shift = self.max_degree - \
                self.permutation_arguments[0].quotient_degree_bound()
            print("verifier shift:", shift)
            terms += [quotient *
                      self.xfield.lift(self.fri.domain(index) ^ shift)]

            difference = (
                processor_point[ProcessorTable.memory_permutation] - memory_point[MemoryTable.permutation])
            quotient = difference / \
                (self.xfield.lift(self.fri.domain(index)) - self.xfield.one())
            terms += [quotient]
            # shift = self.max_degree - \
            #     (BrainfuckStark.roundup_npo2(
            #         self.running_time) + self.num_randomizers - 2)
            shift = self.max_degree - \
                self.permutation_arguments[1].quotient_degree_bound()
            print("verifier shift:", shift)
            terms += [quotient *
                      self.xfield.lift(self.fri.domain(index) ^ shift)]
            print("len(terms) after difference: ", len(terms))

            assert(len(terms) == len(
                weights)), f"length of terms ({len(terms)}) must be equal to length of weights ({len(weights)})"

            # print("verifier terms:")
            # for t in terms:
            #     print(t)
            # print("verifier weights:")
            # for w in weights:
            #     print(w)

            # compute inner product of weights and terms
            inner_product = reduce(
                lambda x, y: x + y, [w * t for w, t in zip(weights, terms)], self.xfield.zero())

            # get value of the combination codeword to test the inner product against
            print("verify read_index: ", proof_stream.read_index)
            combination_leaf = proof_stream.pull()
            print("verify read_index: ", proof_stream.read_index)
            combination_path = proof_stream.pull()
            print("verify read_index: ", proof_stream.read_index)
            print("path:", [hexlify(p) for p in combination_path])

            # verify Merkle authentication path
            print("verifier_verdict:", verifier_verdict)
            verifier_verdict = verifier_verdict and Merkle.verify(
                combination_root, index, combination_path, combination_leaf)
            if not verifier_verdict:
                print(
                    "Merkle authentication path fails for combination codeword value at index", index)
                print("root:", hexlify(combination_root))
                print("index:", index)
                print("path:", [hexlify(p) for p in combination_path])
                print("leaf:", combination_leaf)
                assert(False)
                return False

            # checy equality
            verifier_verdict = verifier_verdict and combination_leaf == inner_product
            if not verifier_verdict:
                print(
                    "inner product does not equal combination codeword element at index", index)
                print("combination leaf: ", combination_leaf)
                print("inner_product: ", inner_product)
                assert(False)
                return False

        # verify low degree of combination polynomial
        print("starting self.fri verification ...")
        tick = time.time()
        polynomial_points = []
        verifier_verdict = self.fri.verify(proof_stream, combination_root)
        polynomial_points.sort(key=lambda iv: iv[0])
        if verifier_verdict == False:
            print("FRI verification failed.")
            return False
        tock = time.time()
        print("FRI verification took", (tock - tick), "seconds")

        # verify external terminals:
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = challenges
        # input
        verifier_verdict = verifier_verdict and processor_input_evaluation_terminal == VirtualMachine.evaluation_terminal(
            [self.xfield(ord(t)) for t in self.input_symbols], gamma)
        assert(verifier_verdict), "processor input evaluation argument failed"
        # output
        # print("type of output symbols:", type(output_symbols),
        #       "and first element:", type(output_symbols[0]))
        verifier_verdict = verifier_verdict and processor_output_evaluation_terminal == VirtualMachine.evaluation_terminal(
            [self.xfield(ord(t)) for t in self.output_symbols], delta)
        assert(verifier_verdict), "processor output evaluation argument failed"
        # program
        print(type(instruction_evaluation_terminal))
        print(type(VirtualMachine.program_evaluation(
            self.program, a, b, c, eta)))
        verifier_verdict = verifier_verdict and instruction_evaluation_terminal == VirtualMachine.program_evaluation(
            self.program, a, b, c, eta)
        assert(verifier_verdict), "instruction program evaluation argument failed"

        return verifier_verdict
