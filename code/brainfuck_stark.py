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
        extension_polynomials = []
        extension_degree_bounds = []
        extension_polynomials += processor_extension.boundary_quotients()
        extension_polynomials += processor_extension.transition_quotients()
        extension_polynomials += processor_extension.terminal_quotients(
            challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[processor_extension.instruction_permutation_terminal])

        extension_polynomials += instruction_extension.boundary_quotients()
        extension_polynomials += instruction_extension.transition_quotients()
        extension_polynomials += instruction_extension.terminal_quotients(
            challenges=[a, b, c, alpha, eta],
            terminals=[instruction_extension.permutation_terminal, instruction_extension.evaluation_terminal])

        extension_polynomials += memory_extension.boundary_quotients()
        extension_polynomials += memory_extension.transition_quotients()
        extension_polynomials += memory_extension.terminal_quotients(
            challenges=[d, e, f, beta], terminals=[memory_extension.permutation_terminal])

        extension_polynomials += input_extension.boundary_quotients()
        extension_polynomials += input_extension.transition_quotients()
        extension_polynomials += input_extension.terminal_quotients(
            challenges=[gamma], terminals=[input_extension.evaluation_terminal])

        extension_polynomials += output_extension.boundary_quotients()
        extension_polynomials += output_extension.transition_quotients()
        extension_polynomials += output_extension.terminal_quotients(
            challenges=[delta], terminals=[output_extension.evaluation_terminal])

        extension_degree_bounds += ProcessorExtension.boundary_quotient_degree_bounds(
            log_time)
        extension_degree_bounds += ProcessorExtension.transition_quotient_degree_bounds(
            log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta])
        extension_degree_bounds += ProcessorExtension.terminal_quotient_degree_bounds(
            log_time,
            challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta],
            terminals=[processor_extension.instruction_permutation_terminal])

        extension_degree_bounds += InstructionExtension.boundary_quotient_degree_bounds(
            log_time)
        extension_degree_bounds += InstructionExtension.transition_quotient_degree_bounds(
            log_time, challenges=[a, b, c, alpha, eta])
        extension_degree_bounds += InstructionExtension.terminal_quotient_degree_bounds(
            log_time, challenges=[a, b, c, alpha, eta],
            terminals=[instruction_extension.permutation_terminal, instruction_extension.evaluation_terminal])

        extension_degree_bounds += MemoryExtension.boundary_quotient_degree_bounds(
            log_time)
        extension_degree_bounds += MemoryExtension.transition_quotient_degree_bounds(
            log_time, challenges=[d, e, f, beta])
        extension_degree_bounds += MemoryExtension.terminal_quotient_degree_bounds(
            log_time, challenges=[d, e, f, beta],  terminals=[memory_extension.permutation_terminal])

        extension_degree_bounds += IOExtension.boundary_quotient_degree_bounds(
            log_input)
        extension_degree_bounds += IOExtension.transition_quotient_degree_bounds(
            log_input, challenges=[gamma])
        extension_degree_bounds += IOExtension.terminal_quotient_degree_bounds(
            log_time, challenges=[gamma], terminals=[input_extension.evaluation_terminal])

        extension_degree_bounds += IOExtension.boundary_quotient_degree_bounds(
            log_output)
        extension_degree_bounds += IOExtension.transition_quotient_degree_bounds(
            log_output, challenges=[delta])
        extension_degree_bounds += IOExtension.terminal_quotient_degree_bounds(
            log_time, challenges=[delta], terminals=[output_extension.evaluation_terminal])

        # ... and equal initial values
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        extension_polynomials += [(processor_extension_polynomials[ProcessorExtension.instruction_permutation] -
                                   instruction_extension_polynomials[InstructionExtension.permutation]) / (X - self.xfield.one())]
        extension_polynomials += [(processor_extension_polynomials[ProcessorExtension.memory_permutation] -
                                   memory_extension_polynomials[MemoryExtension.permutation]) / (X - self.xfield.one())]
        extension_degree_bounds += [(1 << log_time) - 2] * 2
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

        assert(all(p.degree() <= max_degree for p in extension_polynomials)
               ), "transition quotient degrees do not match with expectation"

        # compute terms of nonlinear combination polynomial
        terms = []
        for i in range(len(base_polynomials)):
            terms += [base_polynomials[i]]
            shift = max_degree - base_degree_bounds[i]
            terms += [(X ^ shift) * base_polynomials[i]]
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

        # instantiate helper objects
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

        # get root of table extensions
        extension_root = proof_stream.pull()

        # get terminal values
        processor_instruction_permutation_terminal = proof_stream.pull()
        processor_memory_permutation = proof_stream.pull()
        processor_input_evaluation_terminal = proof_stream.pull()
        processor_output_evaluation_terminal = proof_stream.pull()
        instruction_evaluation_terminal = proof_stream.pull()

        # get weights for nonlinear combination
        num_base_polynomials = ProcessorTable(self.field).width + InstructionTable(
            self.field).width + MemoryTable(self.field).width + IOTable(self.field).width * 2
        num_extension_polynomials = ProcessorExtension(self.field).width + InstructionExtension(
            self.field).width + MemoryExtension(self.field).width + IOExtension(self.field).width * 2 - num_base_polynomials
        num_randomizer_polynomials = 1
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
        extension_leafs = []
        for _ in quadrupled_indices:
            base_leafs += [proof_stream.pull()]
            extension_leafs += [proof_stream.pull()]

        # get authentication paths
        base_paths = []
        extension_paths = []
        for _ in quadrupled_indices:
            base_paths += [proof_stream.pull()]
            extension_paths += [proof_stream.pull()]

        # verify authentication paths
        for qi, (elm, salt), path in zip(quadrupled_indices, base_leafs, base_paths):
            SaltedMerkle.verify(base_root, qi, salt, path, elm)
        for qi, (elm, salt), path in zip(quadrupled_indices, extension_leafs, extension_paths):
            SaltedMerkle.verify(base_root, qi, salt, path, elm)

        # compute degree bounds
        base_degree_bounds = [base_degree_bound] * num_base_polynomials
        extension_degree_bounds = []

        extension_degree_bounds += ProcessorExtension.boundary_quotient_degree_bounds(
            log_time)
        extension_degree_bounds += InstructionExtension.boundary_quotient_degree_bounds(
            log_time)
        extension_degree_bounds += MemoryExtension.boundary_quotient_degree_bounds(
            log_time)
        extension_degree_bounds += IOExtension.boundary_quotient_degree_bounds(
            log_input)
        extension_degree_bounds += IOExtension.boundary_quotient_degree_bounds(
            log_output)

        extension_degree_bounds += ProcessorExtension.transition_quotient_degree_bounds(
            log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta])
        extension_degree_bounds += InstructionExtension.transition_quotient_degree_bounds(
            log_time, challenges=[a, b, c, alpha, eta])
        extension_degree_bounds += MemoryExtension.transition_quotient_degree_bounds(
            log_time, challenges=[d, e, f, beta])
        extension_degree_bounds += IOExtension.transition_quotient_degree_bounds(
            log_input, challenges=[gamma])
        extension_degree_bounds += IOExtension.transition_quotient_degree_bounds(
            log_output, challenges=[delta])

        extension_degree_bounds += ProcessorExtension.terminal_quotient_degree_bounds(log_time, challenges=[
                                                                                      a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[processor_instruction_permutation_terminal])
        extension_degree_bounds += InstructionExtension.terminal_quotient_degree_bounds(log_time, challenges=[
                                                                                        a, b, c, alpha, eta], terminals=[processor_instruction_permutation_terminal, instruction_evaluation_terminal])
        extension_degree_bounds += MemoryExtension.terminal_quotient_degree_bounds(
            log_time, challenges=[d, e, f, beta],  terminals=[processor_memory_permutation_terminal])
        extension_degree_bounds += IOExtension.terminal_quotient_degree_bounds(
            log_time, challenges=[gamma], terminals=[processor_input_evaluation_terminal])
        extension_degree_bounds += IOExtension.terminal_quotient_degree_bounds(
            log_time, challenges=[delta], terminals=[processor_output_evaluation_terminal])

        extension_degree_bounds += [(1 << log_time) - 2] * 2

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
                sum += weights[2*num_base_polynomials + num_randomizer_polynomials + 2*j] * extension_leafs[0][j] + \
                    weights[2*num_base_polynomials + num_randomizer_polynomials +
                            2*j+1] * extension_leafs[0][j] * (omega ^ shiftj)

            verifier_verdict = verifier_verdict and (pv == sum)
            if not verifier_verdict:
                return False

        # verify air constraints
        for i in range(len(quadrupled_indices)-1):
            qi, (base_elm, _), (ext_elm, _) = zip(
                quadrupled_indices, base_leafs, extension_leafs)[i]
            qi_next, (base_elm_next, _), (ext_elm_next, _) = zip(
                quadrupled_indices, base_leafs, extension_leafs)[i+1]
            if qi_next == qi + 1:
                current_index = qi
                next_index = qi + 1 % rounded_trace_length

                # processor
                processor_extension = ProcessorExtension(
                    a, b, c, d, e, f, alpha, beta, gamma, delta)
                processor_table_indices = range(0, ProcessorTable.width())
                processor_extension_indices = range(0,
                                                    ProcessorExtension.width() - ProcessorTable.width())
                processor_constraints_indices = range(0, len(processor_extension.boundary_constraints_ext())
                                                      + len(processor_extension.transition_constraints_ext())
                                                      + len(processor_extension.terminal_constraints_ext()))
                current_point = base_elm[processor_table_indices] + \
                    ext_elm[processor_extension_indices]
                next_point = base_elm_next[processor_table_indices] + \
                    ext_elm_next[processor_extension_indices]
                processor_extension.verify_rows(
                    current_point, next_point, polynomial_values[current_index][processor_constraints_indices])

                # instruction
                instruction_extension = InstructionExtension(
                    a, b, c, alpha, eta)
                instruction_table_indices = range(ProcessorTable.width(),
                                                  ProcessorTable().width + InstructionTable().width)
                instruction_extension_indices = range(ProcessorExtension().width - ProcessorTable().width,
                                                      ProcessorExtension().width - ProcessorTable().width + InstructionExtension().width - InstructionTable().width)
                instruction_constraints_indices = range(1+max(processor_constraints_indices),
                                                        1+max(processor_constraints_indices) + len(instruction_extension.boundary_constraints_ext()) + len(instruction_extension.transition_constraints_ext()) + len(instruction_extension.terminal_constraints_ext()))
                current_point = base_elm[instruction_table_indices] + \
                    ext_elm[instruction_extension_indices]
                next_point = base_elm_next[instruction_table_indices] + \
                    ext_elm_next[instruction_extension_indices]
                instruction_extension.verify_rows(
                    current_point, next_point, polynomial_values[current_index][instruction_constraints_indices])

                # memory
                memory_extension = MemoryExtension(
                    d, e, f, beta)
                memory_table_indices = range(ProcessorTable().width + InstructionTable().width,
                                             ProcessorTable().width + InstructionTable().width + MemoryTable().width)
                memory_extension_indices = range(ProcessorExtension().width - ProcessorTable().width + InstructionExtension().width - InstructionTable().width,
                                                 ProcessorExtension().width - ProcessorTable().width + InstructionExtension().width - InstructionTable().width + MemoryExtension().width - MemoryTable().width)
                memory_constraints_indices = range(1+max(instruction_constraints_indices),
                                                   1+max(instruction_constraints_indices) + len(memory_extension.boundary_constraints_ext()) + len(memory_extension.transition_constraints_ext()) + len(memory_extension.terminal_constraints_ext()))
                current_point = base_elm[memory_table_indices] + \
                    ext_elm[memory_extension_indices]
                next_point = base_elm_next[memory_table_indices] + \
                    ext_elm_next[memory_extension_indices]
                memory_extension.verify_rows(
                    current_point, next_point, polynomial_values[current_index][memory_constraints_indices])

                # input
                input_extension = IOExtension(gamma)
                input_table_indices = range(ProcessorTable().width + InstructionTable().width + MemoryTable().width,
                                            ProcessorTable().width + InstructionTable().width + MemoryTable().width + IOTable().width)
                input_extension_indices = range(ProcessorExtension().width - ProcessorTable().width + InstructionExtension().width - InstructionTable().width + MemoryExtension().width - MemoryTable().width,
                                                ProcessorExtension().width - ProcessorTable().width + InstructionExtension().width - InstructionTable().width + MemoryExtension().width - MemoryTable().width + IOExtension().width - IOTable().width)
                input_constraints_indices = range(1+max(memory_constraints_indices),
                                                  1+max(memory_constraints_indices) + len(input_extension.boundary_constraints_ext()) + len(input_extension.transition_constraints_ext()) + len(input_extension.terminal_constraints_ext()))
                current_point = base_elm[input_table_indices] + \
                    ext_elm[input_extension_indices]
                next_point = base_elm_next[input_table_indices] + \
                    ext_elm_next[input_extension_indices]
                input_extension.verify_rows(
                    current_point, next_point, polynomial_values[current_index][input_constraints_indices])

                # output
                output_extension = IOExtension(delta)
                output_table_indices = range(ProcessorTable().width + InstructionTable().width + MemoryTable().width + IOTable().width,
                                             ProcessorTable().width + InstructionTable().width + MemoryTable().width + IOTable().width + IOTable().width)
                output_extension_indices = range(ProcessorExtension().width - ProcessorTable().width + InstructionExtension().width - InstructionTable().width + MemoryExtension().width - MemoryTable().width + IOExtension().width - IOTable().width,
                                                 ProcessorExtension().width - ProcessorTable().width + InstructionExtension().width - InstructionTable().width + MemoryExtension().width - MemoryTable().width + IOExtension().width - IOTable().width + IOExtension().width - IOTable().width)
                output_constraints_indices = range(1+max(input_constraints_indices),
                                                   1+max(input_constraints_indices) + len(output_extension.boundary_constraints_ext()) + len(output_extension.transition_constraints_ext()) + len(output_extension.terminal_constraints_ext()))
                current_point = base_elm[output_table_indices] + \
                    ext_elm[output_extension_indices]
                next_point = base_elm_next[output_table_indices] + \
                    ext_elm_next[output_extension_indices]
                output_extension.verify_rows(
                    current_point, next_point, polynomial_values[current_index][output_constraints_indices])

            # verify table relations

        return verifier_verdict
