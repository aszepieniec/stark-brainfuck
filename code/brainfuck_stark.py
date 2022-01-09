from fri import *
from instruction_extension import InstructionExtension
from memory_extension import MemoryExtension
from processor_extension import ProcessorExtension
from io_extension import IOExtension
from univariate import *
from multivariate import *
from ntt import *
from functools import reduce
import os

from vm import VirtualMachine


class BrainfuckStark:
    def __init__(self):
        # set parameters
        self.field = BaseField.main()
        self.expansion_factor = 16
        self.num_colinearity_checks = 40
        self.security_level = 160
        assert(self.expansion_factor & (self.expansion_factor - 1)
               == 0), "expansion factor must be a power of 2"
        assert(self.expansion_factor >=
               4), "expansion factor must be 4 or greater"
        assert(self.num_colinearity_checks * len(bin(self.expansion_factor)
               [3:]) >= self.security_level), "number of colinearity checks times log of expansion factor must be at least security level"

        self.num_randomizers = 4*self.num_colinearity_checks

        self.vm = VirtualMachine()

    def transition_degree_bounds(self, transition_constraints):
        point_degrees = [1] + [self.original_trace_length +
                               self.num_randomizers-1] * 2*self.num_registers
        return [max(sum(r*l for r, l in zip(point_degrees, k)) for k, v in a.dictionary.items()) for a in transition_constraints]

    def transition_quotient_degree_bounds(self, transition_constraints):
        return [d - (self.original_trace_length-1) for d in self.transition_degree_bounds(transition_constraints)]

    def max_degree(self, transition_constraints):
        md = max(self.transition_quotient_degree_bounds(transition_constraints))
        return (1 << (len(bin(md)[2:]))) - 1

    def boundary_zerofiers(self, boundary):
        zerofiers = []
        for s in range(self.num_registers):
            points = [self.omicron ^ c for c, r, v in boundary if r == s]
            zerofiers = zerofiers + [Polynomial.zerofier_domain(points)]
        return zerofiers

    def boundary_interpolants(self, boundary):
        interpolants = []
        for s in range(self.num_registers):
            points = [(c, v) for c, r, v in boundary if r == s]
            domain = [self.omicron ^ c for c, v in points]
            values = [v for c, v in points]
            interpolants = interpolants + \
                [Polynomial.interpolate_domain(domain, values)]
        return interpolants

    def boundary_quotient_degree_bounds(self, randomized_trace_length, boundary):
        randomized_trace_degree = randomized_trace_length - 1
        return [randomized_trace_degree - bz.degree() for bz in self.boundary_zerofiers(boundary)]

    def sample_weights(self, number, randomness):
        return [self.field.sample(blake2b(randomness + bytes(i)).digest()) for i in range(0, number)]

    @staticmethod
    def roundup_npo2(integer):
        if integer == 0 or integer == 1:
            return 1
        return 1 << (len(bin(integer-1)[2:]))

    def prove(self, processor_table, instruction_table, memory_table, input_table, output_table, proof_stream=None):
        # infer details about computation
        original_trace_length = len(processor_table.table)
        rounded_trace_length = BrainfuckStark.roundup_npo2(
            original_trace_length)
        randomized_trace_length = rounded_trace_length + self.num_randomizers

        # compute fri domain length
        air_degree = 8  # TODO verify me
        tp_degree = air_degree * (randomized_trace_length - 1)
        tq_degree = tp_degree - (rounded_trace_length - 1)
        max_degree = BrainfuckStark.roundup_npo2(
            tq_degree + 1) - 1  # The max degree bound provable by FRI
        fri_domain_length = (max_degree+1) * self.expansion_factor

        # compute generators
        generator = self.field.generator()
        omega = self.field.primitive_nth_root(fri_domain_length)
        omicron = self.field.primitive_nth_root(
            rounded_trace_length)

        # check numbers for sanity
        # print(original_trace_length)
        # print(rounded_trace_length)
        # print(randomized_trace_length)
        # print(air_degree)
        # print(tp_degree)
        # print(tq_degree)
        # print(tqd_roundup)
        # print(fri_domain_length)

        # instantiate helper objects
        fri = Fri(generator, omega, fri_domain_length,
                  self.expansion_factor, self.num_colinearity_checks)

        if proof_stream == None:
            proof_stream = ProofStream()

        # interpolate with randomization
        randomizer_offset = self.generator^2
        processor_polynomials = processor_table.interpolate(
            randomizer_offset, omicron, rounded_trace_length, self.num_randomizers)
        instruction_polynomials = instruction_table.interpolate(
            randomizer_offset, omicron, rounded_trace_length, self.num_randomizers)
        memory_polynomials = memory_table.interpolate(
            randomizer_offset, omicron, rounded_trace_length, self.num_randomizers)
        input_polynomials = input_table.interpolate(
            randomizer_offset, omicron, rounded_trace_length, self.num_randomizers)
        output_polynomials = output_table.interpolate(
            randomizer_offset, omicron, rounded_trace_length, self.num_randomizers)

        base_polynomials = processor_polynomials + instruction_polynomials + memory_polynomials + input_polynomials + output_polynomials
        base_degree_bounds = [rounded_trace_length + self.num_randomizers - 1] * len(base_polynomials)

        # commit
        base_codewords = [fast_coset_evaluate(p, self.generator, self.omega, self.fri_domain_length) 
            for p in ([processor_polynomials] + [instruction_polynomials] + [memory_polynomials] + [input_polynomials] + [output_polynomials])]
        zipped_codeword = zip(base_codewords)
        base_tree = SaltedMerkle(zipped_codeword)
        proof_stream.push(base_tree.root())

        # get coefficients for table extensions
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = self.sample_weights(11, proof_stream.prover_fiat_shamir())

        # extend tables
        processor_extension = ProcessorExtension.extend(
            processor_table, a, b, c, d, e, f, alpha, beta, gamma, delta)
        instruction_extension = InstructionExtension.extend(
            instruction_table, a, b, c, alpha, eta)
        memory_extension = MemoryExtension.extend(memory_table, d, e, f, beta)
        input_extension = IOExtension.extend(input_table, gamma)
        output_extension = IOExtension.extend(output_table, delta)

        # get terminal values
        processor_instruction_permutation_terminal = processor_extension.instruction_permutation_terminal
        processor_memory_permutation_terminal = processor_extension.memory_permutation_terminal
        processor_input_evaluation_terminal = processor_extension.input_evaluation_terminal
        processor_output_evaluation_terminal = processor_extension.output_evaluation_terminal
        instruction_evaluation_terminal = instruction_extension.evaluation_terminal

        # send terminals
        proof_stream.push(processor_instruction_permutation_terminal)
        proof_stream.push(processor_memory_permutation_terminal)
        proof_stream.push(processor_input_evaluation_terminal)
        proof_stream.push(processor_output_evaluation_terminal)
        proof_stream.push(instruction_evaluation_terminal)

        # interpolate extension columns
        processor_extension_polynomials = processor_extension.interpolate_extension(
            self.generator ^ 2, omicron, rounded_trace_length, self.num_randomizers)
        instruction_extension_polynomials = instruction_extension.interpolate_extension(
            self.generator ^ 2, omicron, rounded_trace_length, self.num_randomizers)
        memory_extension_polynomials = memory_extension.interpolate_extension(
            self.generator ^ 2, omicron, rounded_trace_length, self.num_randomizers)
        input_extension_polynomials = input_extension.interpolate_extension(
            self.generator ^ 2, omicron, rounded_trace_length, self.num_randomizers)
        output_extension_polynomials = output_extension.interpolate_extension(
            self.generator ^ 2, omicron, rounded_trace_length, self.num_randomizers)

        # commit to extension polynomials
        extension_codewords = [fast_coset_evaluate(p, generator, omega, fri_domain_length)
            for p in ([processor_extension_polynomials] + [instruction_extension_polynomials] + [memory_extension_polynomials] + [input_extension_polynomials] + [output_extension_polynomials])]
        zipped_extension_codeword = zip(extension_codewords)
        extension_tree = SaltedMerkle(zipped_extension_codeword)
        proof_stream.push(extension_tree.root())

        # gather polynomials derived from generalized AIR constraints relating to ...
        extension_polynomials = []
        extension_degree_bounds = []
        # ... boundary ...
        extension_polynomials += processor_extension.boundary_quotients()
        extension_polynomials += instruction_extension.boundary_quotients()
        extension_polynomials += memory_extension.boundary_quotients()
        extension_polynomials += input_extension.boundary_quotients()
        extension_polynomials += output_extension.boundary_quotients()
        extension_degree_bounds += processor_extension.boundary_quotient_degree_bounds()
        extension_degree_bounds += instruction_extension.boundary_quotient_degree_bounds()
        extension_degree_bounds += memory_extension.boundary_quotient_degree_bounds()
        extension_degree_bounds += input_extension.boundary_quotient_degree_bounds()
        extension_degree_bounds += output_extension.boundary_quotient_degree_bounds()
        # ... transitions ...
        extension_polynomials += processor_extension.transition_quotients()
        extension_polynomials += instruction_extension.transition_quotients()
        extension_polynomials += memory_extension.transition_quotients()
        extension_polynomials += input_extension.transition_quotients()
        extension_polynomials += output_extension.transition_quotients()
        extension_degree_bounds += processor_extension.transition_quotient_degree_bounds()
        extension_degree_bounds += instruction_extension.transition_quotient_degree_bounds()
        extension_degree_bounds += memory_extension.transition_quotient_degree_bounds()
        extension_degree_bounds += input_extension.transition_quotient_degree_bounds()
        extension_degree_bounds += output_extension.transition_quotient_degree_bounds()
        # ... terminal values ...
        extension_polynomials += processor_extension.terminal_quotients()
        extension_polynomials += instruction_extension.terminal_quotients()
        extension_polynomials += memory_extension.terminal_quotients()
        extension_polynomials += input_extension.terminal_quotients()
        extension_polynomials += output_extension.terminal_quotients()
        extension_degree_bounds += processor_extension.terminal_quotient_degree_bounds()
        extension_degree_bounds += instruction_extension.terminal_quotient_degree_bounds()
        extension_degree_bounds += memory_extension.terminal_quotient_degree_bounds()
        extension_degree_bounds += input_extension.terminal_quotient_degree_bounds()
        extension_degree_bounds += output_extension.terminal_quotient_degree_bounds()
        # ... and equal initial values
        X = Polynomial([self.field.zero(), self.field.one()])
        extension_polynomials += [(processor_extension_polynomials[ProcessorExtension.instruction_permutation] -
                        instruction_extension_polynomials[InstructionExtension.permutation]) / (X - omicron)]
        extension_polynomials += [(processor_extension_polynomials[ProcessorExtension.memory_permutation] -
                        memory_extension_polynomials[MemoryExtension.permutation]) / (X - omicron)]
        extension_degree_bounds += [randomized_trace_length + self.num_randomizers - 2] * 2
        # (don't need to subtract equal values for the io evaluations because they're not randomized)

        # sample randomizer polynomial
        randomizer_polynomial = Polynomial([self.xfield.sample(os.urandom(
            3*9)) for i in range(max_degree+1)])
        randomizer_codeword = fast_coset_evaluate(randomizer_polynomial, self.generator, omega, fri_domain_length)

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every other polynomial
        weights = self.sample_weights(2*len(extension_polynomials) + 1, proof_stream.prover_fiat_shamir())

        assert(all(p.degree() <= max_degree for p in extension_polynomials)), "transition quotient degrees do not match with expectation"

        # compute terms of nonlinear combination polynomial
        terms = []
        for i in range(len(base_polynomials)):
            terms += [base_polynomials[i]]
            shift = max_degree - base_degree_bounds[i]
            terms += [(X^shift) * base_polynomials[i]]
        for i in range(len(extension_polynomials)):
            terms += [extension_polynomials[i]]
            shift = max_degree - extension_degree_bounds[i]
            terms += [(X ^ shift) * extension_polynomials[i]]
        terms += [randomizer_polynomial]

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for all i)
        combination = reduce(
            lambda a, b: a+b, [Polynomial([weights[i]]) * terms[i] for i in range(len(terms))], Polynomial([]))

        # compute matching codeword
        combined_codeword = fast_coset_evaluate(
            combination, self.generator, self.omega, self.fri_domain_length)

        # prove low degree of combination polynomial, and collect indices
        indices = self.fri.prove(combined_codeword, proof_stream)

        # process indices
        duplicated_indices = [i for i in indices] + \
            [(i + self.expansion_factor) %
             self.fri.domain_length for i in indices]
        quadrupled_indices = [i for i in duplicated_indices] + [
            (i + (fri.domain_length // 2)) % fri.domain_length for i in duplicated_indices]
        quadrupled_indices.sort()

        # open indicated positions in all the codewords and in both trees
        for c in base_codewords + extension_codewords + [randomizer_codeword]:
            for i in quadrupled_indices:
                proof_stream.push(c[i])
        paths = [base_tree.open(qi for qi in quadrupled_indices)]
        proof_stream.push(paths)

        # the final proof is just the serialized stream
        return proof_stream.serialize()

    def verify(self, proof, transition_constraints, boundary, transition_zerofier_root, proof_stream=None):
        H = blake2b

        # infer trace length from boundary conditions
        original_trace_length = 1 + max(c for c, r, v in boundary)
        randomized_trace_length = original_trace_length + self.num_randomizers

        # deserialize with right proof stream
        if proof_stream == None:
            proof_stream = ProofStream()
        proof_stream = proof_stream.deserialize(proof)

        # get Merkle roots of boundary quotient codewords
        boundary_quotient_roots = []
        for s in range(self.num_registers):
            boundary_quotient_roots = boundary_quotient_roots + \
                [proof_stream.pull()]

        # get Merkle root of randomizer polynomial
        randomizer_root = proof_stream.pull()

        # get weights for nonlinear combination
        weights = self.sample_weights(1 + 2*len(transition_constraints) + 2*len(
            self.boundary_interpolants(boundary)), proof_stream.verifier_fiat_shamir())

        # verify low degree of combination polynomial
        polynomial_values = []
        verifier_accepts = self.fri.verify(proof_stream, polynomial_values)
        polynomial_values.sort(key=lambda iv: iv[0])
        if not verifier_accepts:
            return False

        indices = [i for i, v in polynomial_values]
        values = [v for i, v in polynomial_values]

        # read and verify leafs, which are elements of boundary quotient codewords
        duplicated_indices = [i for i in indices] + \
            [(i + self.expansion_factor) %
             self.fri.domain_length for i in indices]
        duplicated_indices.sort()
        leafs = []
        for r in range(len(boundary_quotient_roots)):
            leafs = leafs + [dict()]
            for i in duplicated_indices:
                leafs[r][i] = proof_stream.pull()
                path = proof_stream.pull()
                verifier_accepts = verifier_accepts and Merkle.verify(
                    boundary_quotient_roots[r], i, path, leafs[r][i])
                if not verifier_accepts:
                    return False

        # read and verify randomizer leafs
        randomizer = dict()
        for i in duplicated_indices:
            randomizer[i] = proof_stream.pull()
            path = proof_stream.pull()
            verifier_accepts = verifier_accepts and Merkle.verify(
                randomizer_root, i, path, randomizer[i])
            if not verifier_accepts:
                return False

        # read and verify transition zerofier leafs
        transition_zerofier = dict()
        for i in duplicated_indices:
            transition_zerofier[i] = proof_stream.pull()
            path = proof_stream.pull()
            verifier_accepts = verifier_accepts and Merkle.verify(
                transition_zerofier_root, i, path, transition_zerofier[i])
            if not verifier_accepts:
                return False

        # verify leafs of combination polynomial
        for i in range(len(indices)):
            current_index = indices[i]  # do need i

            # get trace values by applying a correction to the boundary quotient values (which are the leafs)
            domain_current_index = self.generator * \
                (self.omega ^ current_index)
            next_index = (current_index +
                          self.expansion_factor) % self.fri.domain_length
            domain_next_index = self.generator * (self.omega ^ next_index)
            current_trace = [self.field.zero()
                             for s in range(self.num_registers)]
            next_trace = [self.field.zero() for s in range(self.num_registers)]
            for s in range(self.num_registers):
                zerofier = self.boundary_zerofiers(boundary)[s]
                interpolant = self.boundary_interpolants(boundary)[s]

                current_trace[s] = leafs[s][current_index] * zerofier.evaluate(
                    domain_current_index) + interpolant.evaluate(domain_current_index)
                next_trace[s] = leafs[s][next_index] * zerofier.evaluate(
                    domain_next_index) + interpolant.evaluate(domain_next_index)

            point = [domain_current_index] + current_trace + next_trace
            transition_constraints_values = [transition_constraints[s].evaluate(
                point) for s in range(len(transition_constraints))]

            # compute nonlinear combination
            counter = 0
            terms = []
            terms += [randomizer[current_index]]
            for s in range(len(transition_constraints_values)):
                tcv = transition_constraints_values[s]
                quotient = tcv / transition_zerofier[current_index]
                terms += [quotient]
                shift = self.max_degree(
                    transition_constraints) - self.transition_quotient_degree_bounds(transition_constraints)[s]
                terms += [quotient * (domain_current_index ^ shift)]
            for s in range(self.num_registers):
                bqv = leafs[s][current_index]  # boundary quotient value
                terms += [bqv]
                shift = self.max_degree(
                    transition_constraints) - self.boundary_quotient_degree_bounds(randomized_trace_length, boundary)[s]
                terms += [bqv * (domain_current_index ^ shift)]
            combination = reduce(
                lambda a, b: a+b, [terms[j] * weights[j] for j in range(len(terms))], self.field.zero())

            # verify against combination polynomial value
            verifier_accepts = verifier_accepts and (combination == values[i])
            if not verifier_accepts:
                return False

        return verifier_accepts
