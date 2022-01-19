from extension_field import ExtensionField
from fri import *
from instruction_extension import InstructionExtension
from instruction_table import InstructionTable
from io_table import IOTable
from memory_extension import MemoryExtension
from memory_table import MemoryTable
from processor_extension import ProcessorExtension
from io_extension import IOExtension
from processor_table import ProcessorTable
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
        self.xfield = ExtensionField.main()
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

        # infer table lengths (=# rows)
        log_time = 1
        while (1 << log_time) < rounded_trace_length:
            log_time += 1
        log_input = 1
        while (1 << log_input) < len(input_table.table):
            log_input += 1
        log_output = 1
        while (1 << log_output) < len(output_table.table):
            log_output += 1

        # compute fri domain length
        air_degree = 10  # TODO verify me
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

        # instantiate helper objects
        fri = Fri(generator, omega, fri_domain_length,
                  self.expansion_factor, self.num_colinearity_checks)

        if proof_stream == None:
            proof_stream = ProofStream()

        # interpolate with randomization
        randomizer_offset = self.generator ^ 2
        processor_polynomials = processor_table.interpolate(
            omega, fri_domain_length, self.num_randomizers)
        instruction_polynomials = instruction_table.interpolate(
            omega, fri_domain_length, self.num_randomizers)
        memory_polynomials = memory_table.interpolate(
            omega, fri_domain_length, self.num_randomizers)
        input_polynomials = input_table.interpolate(
            omega, fri_domain_length, self.num_randomizers)
        output_polynomials = output_table.interpolate(
            omega, fri_domain_length, self.num_randomizers)

        base_polynomials = processor_polynomials + instruction_polynomials + \
            memory_polynomials + input_polynomials + output_polynomials
        base_degree_bound = 1
        while base_degree_bound < rounded_trace_length + self.num_randomizers:
            base_degree_bound = base_degree_bound << 1
        base_degree_bound -= 1
        base_degree_bounds = [base_degree_bound] * len(base_polynomials)

        # sample randomizer polynomial
        randomizer_polynomial = Polynomial([self.xfield.sample(os.urandom(
            3*9)) for i in range(max_degree+1)])
        randomizer_codeword = fast_coset_evaluate(
            randomizer_polynomial, self.generator, omega, fri_domain_length)

        # commit
        base_codewords = [fast_coset_evaluate(p, self.generator, self.omega, self.fri_domain_length)
                          for p in ([processor_polynomials] + [instruction_polynomials] + [memory_polynomials] + [input_polynomials] + [output_polynomials])]
        zipped_codeword = zip(base_codewords + [randomizer_codeword])
        base_tree = SaltedMerkle(zipped_codeword)
        proof_stream.push(base_tree.root())

        # get coefficients for table extensions
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = self.sample_weights(
            11, proof_stream.prover_fiat_shamir())

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
        processor_extension_polynomials = processor_extension.interpolate(
            randomizer_offset, omega, fri_domain_length, self.num_randomizers)
        instruction_extension_polynomials = instruction_extension.interpolate(
            randomizer_offset, omega, fri_domain_length, self.num_randomizers)
        memory_extension_polynomials = memory_extension.interpolate(
            randomizer_offset, omega, fri_domain_length, self.num_randomizers)
        input_extension_polynomials = input_extension.interpolate(
            randomizer_offset, omega, fri_domain_length, self.num_randomizers)
        output_extension_polynomials = output_extension.interpolate(
            randomizer_offset, omega, fri_domain_length, self.num_randomizers)

        # commit to extension polynomials
        extension_codewords = [fast_coset_evaluate(p, generator, omega, fri_domain_length)
                               for p in ([processor_extension_polynomials] + [instruction_extension_polynomials] + [memory_extension_polynomials] + [input_extension_polynomials] + [output_extension_polynomials])]
        zipped_extension_codeword = zip(extension_codewords)
        extension_tree = SaltedMerkle(zipped_extension_codeword)
        proof_stream.push(extension_tree.root())

        # gather polynomials derived from generalized AIR constraints relating to boundary, transition, and terminals
        quotient_polynomials = []
        quotient_polynomials += processor_extension.all_quotients()
        quotient_polynomials += instruction_extension.all_quotients()
        quotient_polynomials += memory_extension.all_quotients()
        quotient_polynomials += input_extension.all_quotients()
        quotient_polynomials += output_extension.all_quotients()

        quotient_degree_bounds = []
        quotient_degree_bounds += processor_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += instruction_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += memory_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += input_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += output_extension.all_quotient_degree_bounds()

        # ... and equal initial values
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        quotient_polynomials += [(processor_extension_polynomials[ProcessorExtension.instruction_permutation] -
                                  instruction_extension_polynomials[InstructionExtension.permutation]) / (X - self.xfield.one())]
        quotient_polynomials += [(processor_extension_polynomials[ProcessorExtension.memory_permutation] -
                                  memory_extension_polynomials[MemoryExtension.permutation]) / (X - self.xfield.one())]
        quotient_degree_bounds += [(1 << log_time) - 2] * 2
        # (don't need to subtract equal values for the io evaluations because they're not randomized)

        # send terminals
        proof_stream.push(processor_extension.instruction_permutation_terminal)
        proof_stream.push(processor_extension.memory_permutation_terminal)
        proof_stream.push(processor_extension.input_evaluation_terminal)
        proof_stream.push(processor_extension.output_evaluation_terminal)
        proof_stream.push(instruction_extension.evaluation)

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every other polynomial (base and extension)
        num_base_polynomials = ProcessorTable(self.field).width + InstructionTable(
            self.field).width + MemoryTable(self.field).width + IOTable(self.field).width * 2
        num_extension_polynomials = ProcessorExtension(self.field).width + InstructionExtension(
            self.field).width + MemoryExtension(self.field).width + IOExtension(self.field).width * 2 - num_base_polynomials
        num_randomizer_polynomials = 1
        weights = self.sample_weights(2*num_base_polynomials + 2*num_extension_polynomials +
                                      num_randomizer_polynomials, proof_stream.prover_fiat_shamir())

        assert(all(p.degree() <= max_degree for p in quotient_polynomials)
               ), "transition quotient degrees do not match with expectation"

        # compute terms of nonlinear combination polynomial
        terms = []
        for i in range(len(base_polynomials)):
            terms += [base_polynomials[i]]
            shift = max_degree - base_degree_bounds[i]
            terms += [(X ^ shift) * base_polynomials[i]]
        for i in range(len(quotient_polynomials)):
            terms += [quotient_polynomials[i]]
            shift = max_degree - quotient_degree_bounds[i]
            terms += [(X ^ shift) * quotient_polynomials[i]]
        terms += [randomizer_polynomial]

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for all i)
        combination = reduce(
            lambda a, b: a+b, [Polynomial([weights[i]]) * terms[i] for i in range(len(terms))], Polynomial([]))

        # compute matching codeword
        combined_codeword = fast_coset_evaluate(
            combination, self.generator, self.omega, self.fri_domain_length)

        # prove low degree of combination polynomial, and collect indices
        indices = fri.prove(combined_codeword, proof_stream)

        # process indices
        duplicated_indices = [i for i in indices] + \
            [(i + self.expansion_factor) %
             fri.domain_length for i in indices]
        quadrupled_indices = [i for i in duplicated_indices] + [
            (i + (fri.domain_length // 2)) % fri.domain_length for i in duplicated_indices]
        quadrupled_indices.sort()

        # open indicated leafs in both trees
        for i in quadrupled_indices:
            proof_stream.push(base_tree.leafs[i])
            proof_stream.push(extension_tree.leafs[i])

        # authenticate indicated leafs in both trees
        for i in quadrupled_indices:
            proof_stream.push(base_tree.open(i))
            proof_stream.push(extension_tree.open(i))

        # the final proof is just the serialized stream
        return proof_stream.serialize()

    def verify(self, proof, log_time, program, input_symbols, output_symbols, proof_stream=None):
        # infer details about computation
        rounded_trace_length = 1 << log_time
        randomized_trace_length = rounded_trace_length + self.num_randomizers
        base_degree_bound = 1
        while base_degree_bound < rounded_trace_length + self.num_randomizers:
            base_degree_bound = base_degree_bound << 1
        base_degree_bound -= 1

        # infer table lengths (=# rows)
        log_input = 1
        while (1 << log_input) < len(input_symbols):
            log_input += 1
        log_output = 1
        while (1 << log_output) < len(output_symbols):
            log_output += 1

        # compute fri domain length
        air_degree = 10  # TODO verify me
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

        # instantiate subprotocol objects
        fri = Fri(generator, omega, fri_domain_length,
                  self.expansion_factor, self.num_colinearity_checks)

        # deserialize with right proof stream
        if proof_stream == None:
            proof_stream = ProofStream()
        proof_stream = proof_stream.deserialize(proof)

        # get Merkle root of base tables
        base_root = proof_stream.pull()

        # get coefficients for table extensions
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = self.sample_weights(
            11, proof_stream.verifier_fiat_shamir())

        # get terminals
        processor_instruction_permutation_terminal = proof_stream.pull()
        processor_memory_permutation_terminal = proof_stream.pull()
        processor_input_evaluation_terminal = proof_stream.pull()
        processor_output_evaluation_terminal = proof_stream.pull()
        instruction_evaluation_terminal = proof_stream.pull()

        # get root of table extensions and quotients
        second_root = proof_stream.pull()

        # get terminal values
        processor_instruction_permutation_terminal = proof_stream.pull()
        processor_memory_permutation_terminal = proof_stream.pull()
        processor_input_evaluation_terminal = proof_stream.pull()
        processor_output_evaluation_terminal = proof_stream.pull()
        instruction_evaluation_terminal = proof_stream.pull()

        # prepare to verify tables
        processor_extension = ProcessorExtension.prepare_verify(log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[
                                                                processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal])
        instruction_extension = InstructionExtension.prepare_verify(log_time, challenges=[a, b, c, alpha, eta], terminals=[
                                                                    processor_instruction_permutation_terminal, instruction_evaluation_terminal])
        memory_extension = MemoryExtension.prepare_verify(log_time, challenges=[
                                                          d, e, f, beta], terminals=[processor_memory_permutation_terminal])
        input_extension = IOExtension.prepare_verify(
            log_input, challenges=[gamma], terminals=[processor_input_evaluation_terminal])
        output_extension = IOExtension.prepare_verify(
            log_output, challenges=[delta], terminals=[processor_output_evaluation_terminal])

        # get weights for nonlinear combination
        num_base_polynomials = ProcessorTable(self.field).width + InstructionTable(
            self.field).width + MemoryTable(self.field).width + IOTable(self.field).width * 2
        num_randomizer_polynomials = 1
        num_extension_polynomials = processor_extension.width + instruction_extension.width + \
            memory_extension.width + input_extension.width + \
            output_extension.width - num_base_polynomials
        num_quotient_polynomials = processor_extension.num_quotients() + instruction_extension.num_quotients() + \
            memory_extension.num_quotients() + input_extension.num_quotients() + \
            output_extension.num_quotients()
        weights = self.sample_weights(2*num_base_polynomials + 2*num_extension_polynomials +
                                      num_randomizer_polynomials, proof_stream.verifier_fiat_shamir())

        # verify low degree of combination polynomial
        polynomial_values = []
        verifier_verdict = fri.verify(proof_stream, polynomial_values)
        polynomial_values.sort(key=lambda iv: iv[0])
        if verifier_verdict == False:
            return False

        indices = [i for i, _ in polynomial_values]
        values = [v for _, v in polynomial_values]

        # process indices
        duplicated_indices = [i for i in indices] + \
            [(i + self.expansion_factor) %
             fri.domain_length for i in indices]
        quadrupled_indices = [i for i in duplicated_indices] + [
            (i + (fri.domain_length // 2)) % fri.domain_length for i in duplicated_indices]
        quadrupled_indices.sort()

        # get leafs
        base_leafs = []
        secondary_leafs = []
        for _ in quadrupled_indices:
            base_leafs += [proof_stream.pull()]
            secondary_leafs += [proof_stream.pull()]

        # get authentication paths
        base_paths = []
        secondary_paths = []
        for _ in quadrupled_indices:
            base_paths += [proof_stream.pull()]
            secondary_paths += [proof_stream.pull()]

        # verify authentication paths
        for qi, (elm, salt), path in zip(quadrupled_indices, base_leafs, base_paths):
            SaltedMerkle.verify(base_root, qi, salt, path, elm)
        for qi, (elm, salt), path in zip(quadrupled_indices, secondary_leafs, secondary_paths):
            SaltedMerkle.verify(base_root, qi, salt, path, elm)

        # compute degree bounds
        base_degree_bounds = [base_degree_bound] * num_base_polynomials
        extension_degree_bounds = [
            base_degree_bound] * num_extension_polynomials
        quotient_degree_bounds = []

        quotient_degree_bounds += processor_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += instruction_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += memory_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += input_extension.all_quotient_degree_bounds()
        quotient_degree_bounds += output_extension.all_quotient_degree_bounds()

        quotient_degree_bounds += [(1 << log_time) - 2] * 2

        # verify nonlinear combination
        for pv in polynomial_values:

            sum = self.xfield.zero()
            for j in range(num_base_polynomials):
                shiftj = max_degree - base_degree_bounds[j]
                sum += weights[2*j] * base_leafs[0][j] + \
                    weights[2*j+1] * base_leafs[0][j] * (omega ^ shiftj)
            for j in range(num_randomizer_polynomials):
                sum += weights[2*num_base_polynomials+j] * \
                    base_leafs[0][2*num_base_polynomials+j]
            for j in range(num_extension_polynomials):
                shiftj = max_degree - extension_degree_bounds[j]
                sum += weights[2*num_base_polynomials + num_randomizer_polynomials + 2*j] * secondary_leafs[0][j] + \
                    weights[2*num_base_polynomials + num_randomizer_polynomials +
                            2*j+1] * secondary_leafs[0][j] * (omega ^ shiftj)
            for j in range(num_quotient_polynomials):
                shiftj = max_degree - quotient_degree_bounds[j]
                sum += weights[2*num_base_polynomials + num_randomizer_polynomials + 2*num_extension_polynomials + 2*j] * secondary_leafs[0][num_extension_polynomials+j] + \
                    weights[2*num_base_polynomials + num_randomizer_polynomials + 2 *
                            num_extension_polynomials + 2*j + 1] * secondary_leafs[0][num_extension_polynomials+j] * (omega ^ shiftj)

            verifier_verdict = verifier_verdict and (pv == sum)
            if not verifier_verdict:
                return False

        # verify air constraints
        for i in range(len(quadrupled_indices)-1):
            qi, (base_elm, _), (sec_elm, _) = zip(
                quadrupled_indices, base_leafs, secondary_leafs)[i]
            qi_next, (base_elm_next, _), (sec_elm_next, _) = zip(
                quadrupled_indices, base_leafs, secondary_leafs)[i+1]

            if qi_next == qi + 1:
                current_index = qi

                point = base_elm + \
                    sec_elm[0:num_extension_polynomials]
                quotients_from_leafs = sec_elm[num_extension_polynomials:]
                shifted_point = base_elm_next + \
                    sec_elm_next[0:num_extension_polynomials]

                # internal airs
                evaluated_quotients = []
                evaluated_quotients += [processor_extension.evaluate_quotients(
                    omicron, omega ^ current_index, point, shifted_point)]
                evaluated_quotients += [instruction_extension.evaluate_quotients(
                    omicron, omega ^ current_index, point, shifted_point)]
                evaluated_quotients += [memory_extension.evaluate_quotients(
                    omicron, omega ^ current_index, point, shifted_point)]
                evaluated_quotients += [input_extension.evaluate_quotients(
                    omicron, omega ^ current_index, point, shifted_point)]
                evaluated_quotients += [output_extension.evaluate_quotients(
                    omicron, omega ^ current_index, point, shifted_point)]

                # table relations
                # X = Polynomial([self.xfield.zero(), self.xfield.one()])
                # quotient_polynomials += [(processor_extension_polynomials[ProcessorExtension.instruction_permutation] -
                #                         instruction_extension_polynomials[InstructionExtension.permutation]) / (X - self.xfield.one())]
                # quotient_polynomials += [(processor_extension_polynomials[ProcessorExtension.memory_permutation] -
                #                         memory_extension_polynomials[MemoryExtension.permutation]) / (X - self.xfield.one())]
                evaluated_quotients += [(sec_elm[ProcessorExtension.instruction_permutation - ProcessorTable().width] - sec_elm
                                         [processor_extension.width - ProcessorTable().width + InstructionExtension.permutation - InstructionTable().width]) / ((omega ^ current_index) - self.xfield.one())]
                evaluated_quotients += [(sec_elm[ProcessorExtension.memory_permutation - ProcessorTable().width] - sec_elm
                                         [processor_extension.width - ProcessorTable().width + instruction_extension.width - InstructionTable().width + MemoryExtension.permutation - MemoryTable().width]) / ((omega ^ current_index) - self.xfield.one())]

                verifier_verdict = verifier_verdict and evaluated_quotients == quotients_from_leafs

        # verify external terminals:
        # input
        verifier_verdict = verifier_verdict and processor_extension.input_evaluation_terminal == VirtualMachine.evaluation_terminal(
            [self.xfield.lift(t) for t in input_symbols], gamma)
        # output
        verifier_verdict = verifier_verdict and processor_extension.input_evaluation_terminal == VirtualMachine.evaluation_terminal(
            [self.xfield.lift(t) for t in output_symbols], delta)
        # program
        verifier_verdict = verifier_verdict and instruction_evaluation_terminal == VirtualMachine.program_evaluation(
            program, a, b, c, eta)

        return verifier_verdict
