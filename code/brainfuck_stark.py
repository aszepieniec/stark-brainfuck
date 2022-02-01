from email.mime import base
from extension_field import ExtensionField, ExtensionFieldElement
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
import time

from vm import VirtualMachine


class BrainfuckStark:
    def __init__(self, generator, extension_field):
        # set parameters
        self.field = generator.field
        self.generator = generator
        self.xfield = extension_field
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

        self.num_randomizers = 4*self.num_colinearity_checks  # for zero-knowledge
        self.num_randomizers = 0  # for speed

        self.vm = VirtualMachine()

    def transition_degree_bounds(self, transition_constraints):
        point_degrees = [1] + [self.original_trace_length +
                               self.num_randomizers-1] * 2*self.num_registers
        return [max(sum(r*l for r, l in zip(point_degrees, k)) for k, v in a.dictionary.items()) for a in transition_constraints]

    def transition_quotient_degree_bounds(self, transition_constraints):
        return [d - (self.original_trace_length-1) for d in self.transition_degree_bounds(transition_constraints)]

    # def max_degree(self, transition_constraints):
    #     md = max(self.transition_quotient_degree_bounds(transition_constraints))
    #     return (1 << (len(bin(md)[2:]))) - 1

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
        return [self.xfield.sample(blake2b(randomness + bytes(i)).digest()) for i in range(number)]

    @staticmethod
    def roundup_npo2(integer):
        if integer == 0 or integer == 1:
            return 1
        return 1 << (len(bin(integer-1)[2:]))

    @staticmethod
    def xntt(poly, omega, order):
        xfield = poly.coefficients[0].field
        field = xfield.polynomial.coefficients[0].field
        # decompose
        coeffs_lists = [poly.coefficients[i][j]
                        for j in range(3) for i in range(1+poly.degree())]
        # pad
        for i in range(len(coeffs_lists)):
            coeffs_lists[i] += [field.zero()] * \
                (order - len(coeffs_lists[i]))
        # ntt
        transformed_lists = [ntt(omega, cl) for cl in coeffs_lists]
        # recompose
        codeword = [ExtensionFieldElement(Polynomial(
            [transformed_lists[i][j] for j in range(3)]), field) for i in range(order)]
        return codeword

    def prove(self, log_time, program, processor_table, instruction_table, memory_table, input_table, output_table, proof_stream=None):
        assert(len(processor_table.table) & (len(processor_table.table)-1)
               == 0), "length of table must be power of two"

        # infer details about computation
        original_trace_length = len(processor_table.table)
        rounded_trace_length = BrainfuckStark.roundup_npo2(
            original_trace_length + len(program))
        randomized_trace_length = rounded_trace_length + self.num_randomizers

        # infer table lengths (=# rows)
        log_instructions = 0
        while (1 << log_instructions) < BrainfuckStark.roundup_npo2(len(instruction_table.table)):
            log_instructions += 1

        log_input = 0
        if len(input_table.table) == 0:
            log_input -= 1
        else:
            while (1 << log_input) < len(input_table.table):
                log_input += 1
        log_output = 0
        if len(output_table.table) == 0:
            log_output -= 1
        else:
            while (1 << log_output) < len(output_table.table):
                log_output += 1

        print("log time:", log_time)
        print("log input length:", log_input)
        print("log output length:", log_output)

        # compute fri domain length
        air_degree = 9  # TODO verify me
        trace_degree = BrainfuckStark.roundup_npo2(randomized_trace_length) - 1
        tp_degree = air_degree * trace_degree
        tz_degree = rounded_trace_length - 1
        tq_degree = tp_degree - tz_degree
        max_degree = BrainfuckStark.roundup_npo2(
            tq_degree + 1) - 1  # The max degree bound provable by FRI
        fri_domain_length = (max_degree+1) * self.expansion_factor

        print("original trace length:", original_trace_length)
        print("rounded trace length:", rounded_trace_length)
        print("randomized trace length:", randomized_trace_length)
        print("trace degree:", trace_degree)
        print("air degree:", air_degree)
        print("transition polynomials degree:", tp_degree)
        print("transition quotients degree:", tq_degree)
        print("transition zerofier degree:", tz_degree)
        print("max degree:", max_degree)
        print("fri domain length:", fri_domain_length)

        # compute generators
        generator = self.field.generator()
        omega = self.field.primitive_nth_root(fri_domain_length)
        omidi = self.field.primitive_nth_root(max_degree+1)
        omicron = self.field.primitive_nth_root(rounded_trace_length)

        # instantiate helper objects
        fri = Fri(generator, omega, fri_domain_length,
                  self.expansion_factor, self.num_colinearity_checks, self.xfield)

        if proof_stream == None:
            proof_stream = ProofStream()

        # pad tables to height 2^k
        processor_table.pad()
        instruction_table.pad()
        memory_table.pad(processor_table)

        instruction_table.test()  # will fail if some AIR does not evaluate to zero

        print("interpolating base tables ...")
        tick = time.time()
        # interpolate with randomization
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
        tock = time.time()
        print("base table interpolation took", (tock - tick), "seconds")

        base_polynomials = processor_polynomials + instruction_polynomials + \
            memory_polynomials + input_polynomials + output_polynomials
        base_degree_bounds = [processor_table.get_height()-1] * ProcessorTable.width + [instruction_table.get_height()-1] * \
            InstructionTable.width + \
            [memory_table.get_height()-1] * MemoryTable.width
        if input_table.get_height() != 0:
            base_degree_bounds += [input_table.get_height()-1] * IOTable.width
        if output_table.get_height() != 0:
            base_degree_bounds += [output_table.get_height()-1] * IOTable.width

        tick = time.time()
        print("sampling randomizer polynomial ...")
        # sample randomizer polynomial
        randomizer_polynomial = Polynomial([self.xfield.sample(os.urandom(
            3*9)) for i in range(max_degree+1)])
        randomizer_codeword = fri.domain.xevaluate(
            randomizer_polynomial)
        tock = time.time()
        print("sampling randomizer polynomial took", (tock - tick), "seconds")

        tick = time.time()
        print("committing to base polynomials ...")
        # commit
        processor_base_codewords = [
            fri.domain.evaluate(p) for p in processor_polynomials]
        instruction_base_codewords = [
            fri.domain.evaluate(p) for p in instruction_polynomials]
        print("instruction polynomial 0:", instruction_polynomials[0])
        memory_base_codewords = [
            fri.domain.evaluate(p) for p in memory_polynomials]
        input_base_codewords = [
            fri.domain.evaluate(p) for p in input_polynomials]
        output_base_codewords = [
            fri.domain.evaluate(p) for p in output_polynomials]
        zipped_codeword = list(zip(processor_base_codewords + instruction_base_codewords +
                               memory_base_codewords + input_base_codewords + output_base_codewords + [randomizer_codeword]))
        base_tree = SaltedMerkle(zipped_codeword)
        proof_stream.push(base_tree.root())
        tock = time.time()
        print("commitment to base polynomials took", (tock - tick), "seconds")

        # get coefficients for table extensions
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = self.sample_weights(
            11, proof_stream.prover_fiat_shamir())

        print("extending ...")
        tick = time.time()
        # extend tables
        processor_extension = ProcessorExtension.extend(
            processor_table, a, b, c, d, e, f, alpha, beta, gamma, delta)
        instruction_extension = InstructionExtension.extend(
            instruction_table, a, b, c, alpha, eta)
        memory_extension = MemoryExtension.extend(memory_table, d, e, f, beta)
        input_extension = IOExtension.extend(input_table, gamma)
        output_extension = IOExtension.extend(output_table, delta)
        tock = time.time()
        print("computing table extensions took", (tock - tick), "seconds")

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

        tick = time.time()
        print("interpolating extensions ...")
        # interpolate extension columns
        processor_extension_polynomials = processor_extension.interpolate_extension(
            omega, fri_domain_length, self.num_randomizers)
        instruction_extension_polynomials = instruction_extension.interpolate_extension(
            omega, fri_domain_length, self.num_randomizers)
        memory_extension_polynomials = memory_extension.interpolate_extension(
            omega, fri_domain_length, self.num_randomizers)
        input_extension_polynomials = input_extension.interpolate_extension(
            omega, fri_domain_length, self.num_randomizers)
        output_extension_polynomials = output_extension.interpolate_extension(
            omega, fri_domain_length, self.num_randomizers)
        tock = time.time()
        print("interpolation of extensions took", (tock - tick), "seconds")

        tick = time.time()
        print("committing to extension polynomials ...")
        # commit to extension polynomials
        processor_extension_codewords = [fri.domain.xevaluate(p)
                                         for p in processor_extension_polynomials]
        instruction_extension_codewords = [fri.domain.xevaluate(p)
                                           for p in instruction_extension_polynomials]
        memory_extension_codewords = [fri.domain.xevaluate(p)
                                      for p in memory_extension_polynomials]
        input_extension_codewords = [fri.domain.xevaluate(p)
                                     for p in input_extension_polynomials]
        output_extension_codewords = [fri.domain.xevaluate(p)
                                      for p in output_extension_polynomials]
        extension_codewords = processor_extension_codewords + instruction_extension_codewords + \
            memory_extension_codewords + input_extension_codewords + output_extension_codewords
        zipped_extension_codeword = list(zip(extension_codewords))
        extension_tree = SaltedMerkle(zipped_extension_codeword)
        proof_stream.push(extension_tree.root())
        tock = time.time()

        extension_degree_bounds = []
        extension_degree_bounds += [processor_extension.get_height()-1] * (
            ProcessorExtension.width - ProcessorTable.width)
        extension_degree_bounds += [instruction_extension.get_height()-1] * (
            InstructionExtension.width - InstructionTable.width)
        extension_degree_bounds += [memory_extension.get_height()-1] * \
            (MemoryExtension.width - MemoryTable.width)
        if input_extension.get_height() != 0:
            extension_degree_bounds += [input_extension.get_height()-1] * \
                (IOExtension.width - IOTable.width)
        if output_extension.get_height() != 0:
            extension_degree_bounds += [output_extension.get_height()-1] * \
                (IOExtension.width - IOTable.width)
        print("commitment to extension polynomials took",
              (tock - tick), "seconds")

        print("first instruction pointeR:",
              processor_extension.table[0][ProcessorExtension.instruction_pointer])

        processor_table.test()
        processor_extension.test()

        instruction_table.test()
        instruction_extension.test()

        memory_table.test()
        memory_extension.test()

        # combine base + extension
        processor_codewords = [[self.xfield.lift(
            c) for c in codeword] for codeword in processor_base_codewords] + processor_extension_codewords
        instruction_codewords = [[self.xfield.lift(
            c) for c in codeword] for codeword in instruction_base_codewords] + instruction_extension_codewords
        memory_codewords = [[self.xfield.lift(
            c) for c in codeword] for codeword in memory_base_codewords] + memory_extension_codewords
        input_codewords = [[self.xfield.lift(
            c) for c in codeword] for codeword in input_base_codewords] + input_extension_codewords
        output_codewords = [[self.xfield.lift(
            c) for c in codeword] for codeword in output_base_codewords] + output_extension_codewords

        tick = time.time()
        print("computing quotients ...")
        # gather polynomials derived from generalized AIR constraints relating to boundary, transition, and terminals
        quotient_codewords = []
        print("processor table:")
        quotient_codewords += processor_extension.all_quotients(fri.domain, processor_codewords, log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[
            processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal])
        print("instruction table:")
        quotient_codewords += instruction_extension.all_quotients(fri.domain, instruction_codewords, log_instructions, challenges=[a, b, c, alpha, eta], terminals=[
            processor_instruction_permutation_terminal, instruction_evaluation_terminal])
        print("memory table:")
        quotient_codewords += memory_extension.all_quotients(fri.domain, memory_codewords, log_time, challenges=[
            d, e, f, beta], terminals=[processor_memory_permutation_terminal])
        print("input table:")
        quotient_codewords += input_extension.all_quotients(fri.domain, input_codewords,
                                                            log_input, challenges=[gamma], terminals=[processor_input_evaluation_terminal])
        print("output table:")
        quotient_codewords += output_extension.all_quotients(fri.domain, output_codewords,
                                                             log_output, challenges=[delta], terminals=[processor_output_evaluation_terminal])

        quotient_degree_bounds = []
        print("number of degree bounds:")
        quotient_degree_bounds += processor_extension.all_quotient_degree_bounds(log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[
            processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal])
        print(len(quotient_degree_bounds))
        quotient_degree_bounds += instruction_extension.all_quotient_degree_bounds(log_time, challenges=[a, b, c, alpha, eta], terminals=[
            processor_instruction_permutation_terminal, instruction_evaluation_terminal])
        print(len(quotient_degree_bounds))
        quotient_degree_bounds += memory_extension.all_quotient_degree_bounds(log_time, challenges=[
            d, e, f, beta], terminals=[processor_memory_permutation_terminal])
        print(len(quotient_degree_bounds))
        quotient_degree_bounds += input_extension.all_quotient_degree_bounds(
            log_input, challenges=[gamma], terminals=[processor_input_evaluation_terminal])
        print(len(quotient_degree_bounds))
        quotient_degree_bounds += output_extension.all_quotient_degree_bounds(
            log_output, challenges=[delta], terminals=[processor_output_evaluation_terminal])
        print(len(quotient_degree_bounds))

        # ... and equal initial values
        # quotient_codewords += [[(processor_codewords[ProcessorExtension.instruction_permutation][i] -
        #                          instruction_codewords[InstructionExtension.permutation][i]) * self.xfield.lift((fri.domain(i) - self.field.one()).inverse()) for i in range(fri.domain.length)]]
        # quotient_codewords += [[(processor_codewords[ProcessorExtension.memory_permutation][i] -
        #                         memory_codewords[MemoryExtension.permutation][i]) * self.xfield.lift((fri.domain(i) - self.field.one()).inverse()) for i in range(fri.domain.length)]]
        # quotient_degree_bounds += [(1 << log_time) - 2] * 2
        # (don't need to subtract equal values for the io evaluations because they're not randomized)
        tock = time.time()
        print("computing quotients took", (tock - tick), "seconds")

        # send terminals
        proof_stream.push(processor_extension.instruction_permutation_terminal)
        proof_stream.push(processor_extension.memory_permutation_terminal)
        proof_stream.push(processor_extension.input_evaluation_terminal)
        proof_stream.push(processor_extension.output_evaluation_terminal)
        proof_stream.push(instruction_extension.evaluation)

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every other polynomial (base, extension, quotients)
        num_base_polynomials = ProcessorTable.width + \
            InstructionTable.width + MemoryTable.width + IOTable.width * 2
        num_extension_polynomials = ProcessorExtension.width + InstructionExtension.width + \
            MemoryExtension.width + IOExtension.width * 2 - num_base_polynomials
        num_randomizer_polynomials = 1
        weights = self.sample_weights(2*num_base_polynomials + 2*num_extension_polynomials +
                                      num_randomizer_polynomials, proof_stream.prover_fiat_shamir())

        polynomials = []
        for i in range(len(quotient_codewords)):
            polynomials += [fri.domain.xinterpolate(quotient_codewords[i])]
            assert(polynomials[i].degree() <=
                   max_degree), f"degree violation for quotient polynomial {i}; max degree: {max_degree}; observed degree: {polynomials[i].degree()}"

        # compute terms of nonlinear combination polynomial
        # TODO: memoize shifted fri domains
        terms = []
        base_codewords = processor_base_codewords + instruction_base_codewords + \
            memory_base_codewords + input_base_codewords + output_base_codewords
        for i in range(len(base_codewords)):
            terms += [[self.xfield.lift(c) for c in base_codewords[i]]]
            shift = max_degree - base_degree_bounds[i]
            print("shift:", shift)
            terms += [[self.xfield.lift((fri.domain(j) ^ shift) * base_codewords[i][j])
                      for j in range(fri.domain.length)]]
        for i in range(len(extension_codewords)):
            terms += [extension_codewords[i]]
            shift = max_degree - extension_degree_bounds[i]
            print("shift':", shift)
            terms += [[self.xfield.lift(fri.domain(j) ^ shift) * extension_codewords[i][j]
                      for j in range(fri.domain.length)]]
        for i in range(len(quotient_codewords)):
            terms += [quotient_codewords[i]]
            shift = max_degree - quotient_degree_bounds[i]
            print("shift\":", shift)
            terms += [[self.xfield.lift(fri.domain(j) ^ shift) * quotient_codewords[i][j]
                      for j in range(fri.domain.length)]]
        terms += [randomizer_codeword]

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for i)
        combination_codeword = reduce(
            lambda lhs, rhs: [l+r for l, r in zip(lhs, rhs)], [[w * e for e in t] for w, t in zip(weights, terms)], [self.xfield.zero()] * fri.domain.length)

        # prove low degree of combination polynomial, and collect indices
        indices = fri.prove(combination_codeword, proof_stream)

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
