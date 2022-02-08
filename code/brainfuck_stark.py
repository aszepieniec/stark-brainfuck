from concurrent.futures import process
from dataclasses import field
from email.mime import base
from platform import processor
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
from salted_merkle import SaltedMerkle
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

    def air_degree(self):
        return 9  # TODO: verify

    def prove(self, running_time, program, processor_table_table, instruction_table_table, memory_table_table, input_table_table, output_table_table, proof_stream=None):
        assert(len(processor_table_table) & (len(processor_table_table)-1)
               == 0), "length of table must be power of two"

        # infer details about computation
        original_trace_length = running_time
        rounded_trace_length = BrainfuckStark.roundup_npo2(
            original_trace_length + len(program))
        randomized_trace_length = rounded_trace_length + self.num_randomizers

        # infer table lengths (=# rows)
        log_time = 0
        while 1 << log_time < len(processor_table_table):
            log_time += 1

        log_instructions = 0
        while 1 << log_instructions < len(instruction_table_table):
            log_instructions += 1

        log_input = 0
        if len(input_table_table) == 0:
            log_input -= 1
        else:
            while (1 << log_input) < len(input_table_table):
                log_input += 1
        log_output = 0
        if len(output_table_table) == 0:
            log_output -= 1
        else:
            while (1 << log_output) < len(output_table_table):
                log_output += 1

        # print("log time:", log_time)
        # print("log input length:", log_input)
        # print("log output length:", log_output)

        # compute fri domain length
        air_degree = self.air_degree()
        trace_degree = BrainfuckStark.roundup_npo2(randomized_trace_length) - 1
        tp_degree = air_degree * trace_degree
        tz_degree = rounded_trace_length - 1
        tq_degree = tp_degree - tz_degree
        max_degree = BrainfuckStark.roundup_npo2(
            tq_degree + 1) - 1  # The max degree bound provable by FRI
        fri_domain_length = (max_degree+1) * self.expansion_factor

        # print("original trace length:", original_trace_length)
        # print("rounded trace length:", rounded_trace_length)
        # print("randomized trace length:", randomized_trace_length)
        # print("trace degree:", trace_degree)
        # print("air degree:", air_degree)
        # print("transition polynomials degree:", tp_degree)
        # print("transition quotients degree:", tq_degree)
        # print("transition zerofier degree:", tz_degree)
        # print("max degree:", max_degree)
        # print("fri domain length:", fri_domain_length)

        # compute generators
        generator = self.field.generator()
        omega = self.field.primitive_nth_root(fri_domain_length)
        # omidi = self.field.primitive_nth_root(max_degree+1)
        # omicron = self.field.primitive_nth_root(rounded_trace_length)

        # instantiate helper objects
        fri = Fri(generator, omega, fri_domain_length,
                  self.expansion_factor, self.num_colinearity_checks, self.xfield)

        if proof_stream == None:
            proof_stream = ProofStream()

        processor_table = ProcessorTable(self.field, len(
            processor_table_table), omega, fri_domain_length)
        processor_table.table = processor_table_table
        instruction_table = InstructionTable(self.field, len(
            instruction_table_table), omega, fri_domain_length)
        instruction_table.table = instruction_table_table
        memory_table = MemoryTable(self.field, len(
            memory_table_table), omega, fri_domain_length)
        memory_table.table = memory_table_table
        input_table = IOTable(self.field, len(
            input_table_table), omega, fri_domain_length)
        input_table.table = input_table_table
        output_table = IOTable(self.field, len(
            output_table_table), omega, fri_domain_length)
        output_table.table = output_table_table

        # pad tables to height 2^k
        processor_table.pad()
        instruction_table.pad()
        memory_table.pad(processor_table)

        # instruction_table.test()  # will fail if some AIR does not evaluate to zero

        # print("interpolating base tables ...")
        # tick = time.time()
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
        # tock = time.time()
        # print("base table interpolation took", (tock - tick), "seconds")

        # base_polynomials = processor_polynomials + instruction_polynomials + \
        #      memory_polynomials + input_polynomials + output_polynomials
        base_degree_bounds = [processor_table.get_height()-1] * ProcessorTable.width + [instruction_table.get_height()-1] * \
            InstructionTable.width + \
            [memory_table.get_height()-1] * MemoryTable.width
        if input_table.get_height() != 0:
            base_degree_bounds += [input_table.get_height()-1] * IOTable.width
        if output_table.get_height() != 0:
            base_degree_bounds += [output_table.get_height()-1] * IOTable.width

        # tick = time.time()
        # print("sampling randomizer polynomial ...")
        # sample randomizer polynomial
        randomizer_polynomial = Polynomial([self.xfield.sample(os.urandom(
            3*9)) for i in range(max_degree+1)])
        randomizer_codeword = fri.domain.xevaluate(
            randomizer_polynomial)
        # tock = time.time()
        # print("sampling randomizer polynomial took", (tock - tick), "seconds")

        # tick = time.time()
        # print("committing to base polynomials ...")
        # commit
        processor_base_codewords = [
            fri.domain.evaluate(p) for p in processor_polynomials]
        instruction_base_codewords = [
            fri.domain.evaluate(p) for p in instruction_polynomials]
        memory_base_codewords = [
            fri.domain.evaluate(p) for p in memory_polynomials]
        input_base_codewords = [
            fri.domain.evaluate(p) for p in input_polynomials]
        output_base_codewords = [
            fri.domain.evaluate(p) for p in output_polynomials]

        all_base_codewords = [randomizer_codeword] + processor_base_codewords + instruction_base_codewords + \
            memory_base_codewords + input_base_codewords + \
            output_base_codewords

        zipped_codeword = list(zip(*all_base_codewords))
        base_tree = SaltedMerkle(zipped_codeword)
        proof_stream.push(base_tree.root())
        print("-> base tree root:", hexlify(base_tree.root()))
        # tock = time.time()
        # print("commitment to base polynomials took", (tock - tick), "seconds")

        # get coefficients for table extensions
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = self.sample_weights(
            11, proof_stream.prover_fiat_shamir())
        print("** challenges for extension")

        # print("extending ...")
        # tick = time.time()
        # extend tables
        processor_extension = ProcessorExtension.extend(
            processor_table, a, b, c, d, e, f, alpha, beta, gamma, delta)
        instruction_extension = InstructionExtension.extend(
            instruction_table, a, b, c, alpha, eta)
        memory_extension = MemoryExtension.extend(memory_table, d, e, f, beta)
        input_extension = IOExtension.extend(input_table, gamma)
        output_extension = IOExtension.extend(output_table, delta)
        tock = time.time()
        # print("computing table extensions took", (tock - tick), "seconds")

        # get terminal values
        processor_instruction_permutation_terminal = processor_extension.instruction_permutation_terminal
        processor_memory_permutation_terminal = processor_extension.memory_permutation_terminal
        processor_input_evaluation_terminal = processor_extension.input_evaluation_terminal
        processor_output_evaluation_terminal = processor_extension.output_evaluation_terminal
        instruction_evaluation_terminal = instruction_extension.evaluation_terminal

        # tick = time.time()
        # print("interpolating extensions ...")
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
        # tock = time.time()
        # print("interpolation of extensions took", (tock - tick), "seconds")

        # tick = time.time()
        # print("committing to extension polynomials ...")
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
        zipped_extension_codeword = list(zip(*extension_codewords))
        extension_tree = SaltedMerkle(zipped_extension_codeword)
        proof_stream.push(extension_tree.root())
        print("-> extension tree root:", hexlify(extension_tree.root()))
        # tock = time.time()

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
        # print("commitment to extension polynomials took",
            #   (tock - tick), "seconds")

        # processor_table.test()
        # processor_extension.test()

        # instruction_table.test()
        # instruction_extension.test()

        # memory_table.test()
        # memory_extension.test()

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

        # tick = time.time()
        # print("computing quotients ...")
        # gather polynomials derived from generalized AIR constraints relating to boundary, transition, and terminals
        quotient_codewords = []
        # print("processor table:")
        quotient_codewords += processor_extension.all_quotients(fri.domain, processor_codewords, log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[
            processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal])
        # print("instruction table:")
        quotient_codewords += instruction_extension.all_quotients(fri.domain, instruction_codewords, log_instructions, challenges=[a, b, c, alpha, eta], terminals=[
            processor_instruction_permutation_terminal, instruction_evaluation_terminal])
        # print("memory table:")
        quotient_codewords += memory_extension.all_quotients(fri.domain, memory_codewords, log_time, challenges=[
            d, e, f, beta], terminals=[processor_memory_permutation_terminal])
        # print("input table:")
        quotient_codewords += input_extension.all_quotients(fri.domain, input_codewords,
                                                            log_input, challenges=[gamma], terminals=[processor_input_evaluation_terminal])
        # print("output table:")
        quotient_codewords += output_extension.all_quotients(fri.domain, output_codewords,
                                                             log_output, challenges=[delta], terminals=[processor_output_evaluation_terminal])

        quotient_degree_bounds = []
        # print("number of degree bounds:")
        quotient_degree_bounds += processor_extension.all_quotient_degree_bounds(log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[
            processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal])
        print(len(quotient_degree_bounds))
        quotient_degree_bounds += instruction_extension.all_quotient_degree_bounds(log_instructions, challenges=[a, b, c, alpha, eta], terminals=[
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
        # Append the difference quotients
        quotient_codewords += [[(processor_codewords[ProcessorExtension.instruction_permutation][i] -
                                 instruction_codewords[InstructionExtension.permutation][i]) * self.xfield.lift((fri.domain(i) - self.field.one()).inverse()) for i in range(fri.domain.length)]]
        quotient_codewords += [[(processor_codewords[ProcessorExtension.memory_permutation][i] -
                                memory_codewords[MemoryExtension.permutation][i]) * self.xfield.lift((fri.domain(i) - self.field.one()).inverse()) for i in range(fri.domain.length)]]
        quotient_degree_bounds += [(1 << log_instructions) -
                                   2, (1 << log_time) - 2]
        # (don't need to subtract equal values for the io evaluations because they're not randomized)
        # (but we do need to assert their correctness)
        # assert(fri.domain.xinterpolate(quotient_codewords[-2]).degree(
        # ) <= quotient_degree_bounds[-2]), "difference quotient 0: bound not satisfied"
        # assert(fri.domain.xinterpolate(quotient_codewords[-1]).degree(
        # ) <= quotient_degree_bounds[-1]), "difference quotient 1: bound not satisfied"
        # tock = time.time()
        # print("computing quotients took", (tock - tick), "seconds")

        # send terminals
        proof_stream.push(processor_extension.instruction_permutation_terminal)
        proof_stream.push(processor_extension.memory_permutation_terminal)
        proof_stream.push(processor_extension.input_evaluation_terminal)
        proof_stream.push(processor_extension.output_evaluation_terminal)
        proof_stream.push(instruction_extension.evaluation_terminal)
        print("-> processor instruction permutation terminal:",
              processor_extension.instruction_permutation_terminal)
        print("-> processor memory permutation terminal",
              processor_extension.memory_permutation_terminal)
        print("-> processor input permutation terminal",
              processor_extension.input_evaluation_terminal)
        print("-> processor output permutation terminal",
              processor_extension.output_evaluation_terminal)
        print("-> instruction program evaluation terminal",
              instruction_extension.evaluation_terminal)

        # get weights for nonlinear combination
        #  - 1 for randomizer polynomials
        #  - 2 for every other polynomial (base, extension, quotients)
        num_base_polynomials = ProcessorTable.width + \
            InstructionTable.width + MemoryTable.width
        if len(input_table.table) != 0:
            num_base_polynomials += 1
        if len(output_table.table) != 0:
            num_base_polynomials += 1
        num_extension_polynomials = ProcessorExtension.width + InstructionExtension.width + \
            MemoryExtension.width - num_base_polynomials
        if len(input_table.table) != 0:
            num_extension_polynomials += 1
        if len(output_table.table) != 0:
            num_extension_polynomials += 1
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

        # polynomials = []
        # for i in range(len(quotient_codewords)):
        #     polynomials += [fri.domain.xinterpolate(quotient_codewords[i])]
        #     assert(polynomials[i].degree() <=
        #            max_degree), f"degree violation for quotient polynomial {i}; max degree: {max_degree}; observed degree: {polynomials[i].degree()}"

        # compute terms of nonlinear combination polynomial
        # TODO: memoize shifted fri domains
        # print("computing nonlinear combination ...")
        # tick = time.time()
        terms = [randomizer_codeword]
        base_codewords = processor_base_codewords + instruction_base_codewords + \
            memory_base_codewords + input_base_codewords + output_base_codewords
        assert(len(base_codewords) ==
               num_base_polynomials), f"number of base codewords {len(base_codewords)} codewords =/= number of base polynomials f{num_base_polynomials}!"
        for i in range(len(base_codewords)):
            terms += [[self.xfield.lift(c) for c in base_codewords[i]]]
            shift = max_degree - base_degree_bounds[i]
            terms += [[self.xfield.lift((fri.domain(j) ^ shift) * base_codewords[i][j])
                      for j in range(fri.domain.length)]]
        assert(len(extension_codewords) ==
               num_extension_polynomials), f"number of extension codewords {len(extension_codewords)} =/= number of extension polynomials f{num_extension_polynomials}"
        for i in range(len(extension_codewords)):
            terms += [extension_codewords[i]]
            shift = max_degree - extension_degree_bounds[i]
            terms += [[self.xfield.lift(fri.domain(j) ^ shift) * extension_codewords[i][j]
                      for j in range(fri.domain.length)]]
        assert(len(quotient_codewords) ==
               num_quotient_polynomials), f"number of quotient codewords {len(quotient_codewords)} =/= number of quotient polynomials {num_quotient_polynomials}"
        for i in range(len(quotient_codewords)):
            terms += [quotient_codewords[i]]
            shift = max_degree - quotient_degree_bounds[i]
            terms += [[self.xfield.lift(fri.domain(j) ^ shift) * quotient_codewords[i][j]
                      for j in range(fri.domain.length)]]
        # print("got terms after", (time.time() - tick), "seconds")

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for i)
        assert(len(terms) == len(
            weights)), f"number of terms {len(terms)} is not equal to number of weights {len(weights)}"
        combination_codeword = reduce(
            lambda lhs, rhs: [l+r for l, r in zip(lhs, rhs)], [[w * e for e in t] for w, t in zip(weights, terms)], [self.xfield.zero()] * fri.domain.length)
        # print("finished computing nonlinear combination; calculation took", time.time() - tick, "seconds")

        # commit to combination codeword
        combination_tree = Merkle(combination_codeword)
        proof_stream.push(combination_tree.root())
        print("-> combination codeword tree root")

        # get indices of leafs to prove nonlinear combination
        indices_seed = proof_stream.prover_fiat_shamir()
        print("** indices for nonlicombo")
        indices = BrainfuckStark.sample_indices(
            self.security_level, indices_seed, fri.domain.length)

        indices = [0]  # TODO remove me when not debugging

        unit_distances = [table.unit_distance(fri.domain.length) for table in [
            processor_table, instruction_table, memory_table, input_table, output_table]]
        unit_distances = list(set(unit_distances))

        # open leafs of zipped codewords at indicated positions
        for index in indices:
            print("I think the index is", index)
            for distance in [0] + unit_distances:
                next_index = (index + distance) % fri.domain.length
                print("I think next_index is", next_index)
                proof_stream.push(base_tree.leafs[next_index][0])
                proof_stream.push(base_tree.open(next_index))
                proof_stream.push(extension_tree.leafs[next_index][0])
                proof_stream.push(extension_tree.open(next_index))
                print("-> leafs and path for index", index, "+",
                      distance, "=", next_index, "mod", fri.domain.length)

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
        print("starting FRI")
        # TODO: update to drop first codeword
        indices = fri.prove(combination_codeword, proof_stream)
        tock = time.time()
        print("FRI took ", (tock - tick), "seconds")

        # deprecate: use FRI indices to prove nonlicombo

        # tick = time.time()
        # print("opening top level leafs")
        # # process indices
        # duplicated_indices = [i for i in indices] + \
        #     [(i + fri.domain.length) %
        #      fri.domain.length for i in indices]
        # quadrupled_indices = [i for i in duplicated_indices] + [
        #     (i + (fri.domain.length // 2)) % fri.domain.length for i in duplicated_indices]
        # quadrupled_indices.sort()

        # # open indicated leafs in both trees
        # for i in quadrupled_indices:
        #     proof_stream.push(base_tree.leafs[i])
        #     print("-> base tree leaf", i)
        #     proof_stream.push(extension_tree.leafs[i])
        #     print("-> extension tree leaf", i)

        # # authenticate indicated leafs in both trees
        # for i in quadrupled_indices:
        #     proof_stream.push(base_tree.open(i))
        #     print("-> base tree path", i)
        #     proof_stream.push(extension_tree.open(i))
        #     print("-> extension tree path", i)

        # tock = time.time()
        # print("opening top level leafs took", (tock - tick), "seconds")

        # the final proof is just the serialized stream
        return proof_stream.serialize()

    def verify(self, proof, running_time, program, input_symbols, output_symbols, proof_stream=None):
        print("inside verifier \\o/")

        verifier_verdict = True

        # infer details about computation
        log_time = 0
        while 1 << log_time < running_time:
            log_time += 1

        log_instructions = 0
        while 1 << log_instructions < running_time + len(program):
            log_instructions += 1

        original_trace_length = running_time
        rounded_trace_length = BrainfuckStark.roundup_npo2(
            original_trace_length + len(program))
        randomized_trace_length = rounded_trace_length + self.num_randomizers

        # infer table lengths (=# rows)
        log_input = 0
        if len(input_symbols) == 0:
            log_input -= 1
        else:
            while (1 << log_input) < len(input_symbols):
                log_input += 1
        log_output = 0
        if len(output_symbols) == 0:
            log_output -= 1
        else:
            while (1 << log_output) < len(output_symbols):
                log_output += 1

        print("log time:", log_time)
        print("log input length:", log_input)
        print("log output length:", log_output)

        # compute fri domain length
        air_degree = self.air_degree()
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
        # omidi = self.field.primitive_nth_root(max_degree+1)
        # omicron = self.field.primitive_nth_root(rounded_trace_length)

        # instantiate helper objects
        fri = Fri(generator, omega, fri_domain_length,
                  self.expansion_factor, self.num_colinearity_checks, self.xfield)

        # deserialize with right proof stream
        if proof_stream == None:
            proof_stream = ProofStream()
        proof_stream = proof_stream.deserialize(proof)

        # get Merkle root of base tables
        base_root = proof_stream.pull()
        print("<- base tree root:", hexlify(base_root))

        # get matching degree bounds
        # base_polynomials = processor_polynomials + instruction_polynomials + \
        #      memory_polynomials + input_polynomials + output_polynomials
        base_degree_bounds = [BrainfuckStark.roundup_npo2(running_time)-1] * ProcessorTable.width + [BrainfuckStark.roundup_npo2(running_time+len(program))-1] * \
            InstructionTable.width + \
            [BrainfuckStark.roundup_npo2(running_time)-1] * MemoryTable.width
        if len(input_symbols) != 0:
            base_degree_bounds += [len(input_symbols)-1] * IOTable.width
        if len(output_symbols) != 0:
            base_degree_bounds += [len(output_symbols)-1] * IOTable.width

        # get coefficients for table extensions
        a, b, c, d, e, f, alpha, beta, gamma, delta, eta = self.sample_weights(
            11, proof_stream.verifier_fiat_shamir())
        print("** challenges for extension coefficients")

        # get root of table extensions
        extension_root = proof_stream.pull()
        print("<- extension tree root:", hexlify(extension_root))
        # get terminals
        processor_instruction_permutation_terminal = proof_stream.pull()
        print("<- processor instruction permutation terminal:",
              processor_instruction_permutation_terminal)
        processor_memory_permutation_terminal = proof_stream.pull()
        print("<- processor memory permutation terminal:",
              processor_memory_permutation_terminal)
        processor_input_evaluation_terminal = proof_stream.pull()
        print("<- processor input evaluation terminal:",
              processor_input_evaluation_terminal)
        processor_output_evaluation_terminal = proof_stream.pull()
        print("<- processor output evaluation terminal:",
              processor_output_evaluation_terminal)
        instruction_evaluation_terminal = proof_stream.pull()
        print("<- instruction evaluation terminal:",
              instruction_evaluation_terminal)

        # generate extension tables for type information
        # i.e., do not populate tables
        processor_extension = ProcessorExtension(running_time, omega, fri_domain_length,
                                                 a, b, c, d, e, f, alpha, beta, gamma, delta, processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal)
        instruction_extension = InstructionExtension(
            running_time+len(program), omega, fri_domain_length, a, b, c, alpha, eta, processor_instruction_permutation_terminal, instruction_evaluation_terminal)
        memory_extension = MemoryExtension(
            running_time, omega, fri_domain_length, d, e, f, beta, processor_memory_permutation_terminal)
        input_extension = IOExtension(
            len(input_symbols), omega, fri_domain_length, gamma, processor_input_evaluation_terminal)
        output_extension = IOExtension(
            len(output_symbols), omega, fri_domain_length, delta, processor_output_evaluation_terminal)

        extension_degree_bounds = [BrainfuckStark.roundup_npo2(processor_extension.get_height())-1] * (ProcessorExtension.width - ProcessorTable.width) + [BrainfuckStark.roundup_npo2(instruction_extension.get_height())-1] * (InstructionExtension.width - InstructionTable.width) + [
            BrainfuckStark.roundup_npo2(memory_extension.get_height())-1] * (MemoryExtension.width - MemoryTable.width) + [BrainfuckStark.roundup_npo2(input_extension.get_height())-1] * (IOExtension.width - IOTable.width) + [BrainfuckStark.roundup_npo2(output_extension.get_height())-1] * (IOExtension.width - IOTable.width)

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every other polynomial (base, extension, quotients)
        num_base_polynomials = ProcessorTable.width + \
            InstructionTable.width + MemoryTable.width
        if len(input_symbols) > 0:
            num_base_polynomials += IOTable.width
        if len(output_symbols) > 0:
            num_base_polynomials += IOTable.width
        num_extension_polynomials = ProcessorExtension.width + InstructionExtension.width + \
            MemoryExtension.width - num_base_polynomials
        if len(input_symbols) > 0:
            num_extension_polynomials += IOExtension.width - IOTable.width
        if len(output_symbols) > 0:
            num_extension_polynomials += IOExtension.width - IOTable.width
        num_randomizer_polynomials = 1

        num_quotient_polynomials = processor_extension.num_quotients() + instruction_extension.num_quotients() + \
            memory_extension.num_quotients() + input_extension.num_quotients() + \
            output_extension.num_quotients()

        num_difference_quotients = 2

        weights_seed = proof_stream.verifier_fiat_shamir()
        weights = self.sample_weights(
            2*num_base_polynomials +
            2*num_extension_polynomials +
            num_randomizer_polynomials +
            2*num_quotient_polynomials +
            2*num_difference_quotients,
            weights_seed)

        print("** challenges for weights")

        # # prepare to verify tables
        # processor_extension = ProcessorExtension.prepare_verify(log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[
        #     processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal])
        # instruction_extension = InstructionExtension.prepare_verify(log_time, challenges=[a, b, c, alpha, eta], terminals=[
        #     processor_instruction_permutation_terminal, instruction_evaluation_terminal])
        # memory_extension = MemoryExtension.prepare_verify(log_time, challenges=[
        #     d, e, f, beta], terminals=[processor_memory_permutation_terminal])
        # input_extension = IOExtension.prepare_verify(
        #     log_input, challenges=[gamma], terminals=[processor_input_evaluation_terminal])
        # output_extension = IOExtension.prepare_verify(
        #     log_output, challenges=[delta], terminals=[processor_output_evaluation_terminal])

        # # get weights for nonlinear combination
        # num_base_polynomials = ProcessorTable(self.field).width + InstructionTable(
        #     self.field).width + MemoryTable(self.field).width + IOTable(self.field).width * 2
        # num_randomizer_polynomials = 1
        # num_extension_polynomials = processor_extension.width + instruction_extension.width + \
        #     memory_extension.width + input_extension.width + \
        #     output_extension.width - num_base_polynomials
        # num_quotient_polynomials = processor_extension.num_quotients() + instruction_extension.num_quotients() + \
        #     memory_extension.num_quotients() + input_extension.num_quotients() + \
        #     output_extension.num_quotients()
        # weights = self.sample_weights(2*num_base_polynomials + 2*num_extension_polynomials +
        #                               num_randomizer_polynomials, proof_stream.verifier_fiat_shamir())

        # pull Merkle root of combination codeword
        combination_root = proof_stream.pull()
        print("<- combination codeword root")

        # get indices of leafs to verify nonlinear combinatoin
        indices_seed = proof_stream.verifier_fiat_shamir()
        print("** indices for nonlicombo")
        indices = BrainfuckStark.sample_indices(
            self.security_level, indices_seed, fri.domain.length)

        indices = [0]  # TODO remove me when not debugging

        unit_distances = [table.unit_distance(fri.domain.length) for table in [
            processor_extension, instruction_extension, memory_extension, input_extension, output_extension]]
        unit_distances = list(set(unit_distances))

        # get leafs at indicated positions
        tuples = dict()
        for index in indices:
            for distance in [0] + unit_distances:
                idx = (index + distance) % fri.domain.length

                element = proof_stream.pull()
                salt, path = proof_stream.pull()
                verifier_verdict = verifier_verdict and SaltedMerkle.verify(
                    base_root, idx, salt, path, element)
                tuples[idx] = [self.xfield.lift(e) for e in list(element)]
                assert(verifier_verdict), "salted base tree verify must success"

                element = proof_stream.pull()
                salt, path = proof_stream.pull()
                verifier_verdict = verifier_verdict and SaltedMerkle.verify(
                    extension_root, idx, salt, path, element)
                tuples[idx] = tuples[idx] + list(element)
                assert(verifier_verdict), "salted base tree verify must success"

                print("<- leafs and path for index", index, "+",
                      distance, "=", idx, "mod", fri.domain.length)

        assert(num_base_polynomials == len(base_degree_bounds)
               ), f"number of base polynomials {num_base_polynomials} =/= number of base degree bounds {len(base_degree_bounds)}"
        # verify nonlinear combination
        for index in indices:
            # collect terms: randomizer
            terms: list[ExtensionFieldElement] = tuples[index][0:num_randomizer_polynomials]

            # collect terms: base
            for i in range(num_randomizer_polynomials, num_randomizer_polynomials+num_base_polynomials):
                terms += [tuples[index][i]]
                shift = max_degree - \
                    base_degree_bounds[i-num_randomizer_polynomials]
                terms += [tuples[index][i] *
                          self.xfield.lift(fri.domain(index) ^ shift)]

            # collect terms: extension
            extension_offset = num_randomizer_polynomials+ProcessorTable.width + \
                InstructionTable.width+MemoryTable.width
            if len(input_symbols) > 0:
                extension_offset += IOTable.width
            if len(output_symbols) > 0:
                extension_offset += IOTable.width

            assert(len(
                terms) == 2 * extension_offset - num_randomizer_polynomials), f"number of terms {len(terms)} does not match with extension offset {2 * extension_offset - num_randomizer_polynomials}"

            for i in range(num_extension_polynomials):
                print("keys of tuples:", [k for k in tuples.keys()])
                print("extension offset plus i = ", extension_offset + i)
                print("i:", i)
                print("index:", index)
                print("len of tuples at index:", len(tuples[index]))
                terms += [tuples[index][extension_offset+i]]
                shift = max_degree - extension_degree_bounds[i]
                print("extension degree shift: ", shift)
                terms += [tuples[index][extension_offset+i]
                          * self.xfield.lift(fri.domain(index) ^ shift)]

            # collect terms: quotients
            # quotients need to be computed
            acc_index = num_randomizer_polynomials
            processor_point = tuples[index][acc_index:(
                acc_index+ProcessorTable.width)]
            acc_index += ProcessorTable.width
            instruction_point = tuples[index][acc_index:(
                acc_index+InstructionTable.width)]
            acc_index += InstructionTable.width
            memory_point = tuples[index][(acc_index):(
                acc_index+MemoryTable.width)]
            acc_index += MemoryTable.width
            if len(input_symbols) == 0:
                input_point = []
            else:
                input_point = tuples[index][(acc_index):(
                    acc_index+IOTable.width)]
                acc_index += IOTable.width
            if len(output_symbols) == 0:
                output_point = []
            else:
                output_point = tuples[index][(acc_index):(
                    acc_index+IOTable.width)]
                acc_index += IOTable.width
            assert(acc_index == extension_offset,
                   "Column count in verifier must match until extension columns")
            acc_index = extension_offset  # Should be unchanged!!
            processor_point += tuples[index][acc_index:(
                acc_index+ProcessorExtension.width-ProcessorTable.width)]
            acc_index += ProcessorExtension.width-ProcessorTable.width
            instruction_point += tuples[index][(acc_index):(
                acc_index+InstructionExtension.width-InstructionTable.width)]
            acc_index += InstructionExtension.width-InstructionTable.width
            memory_point += tuples[index][(acc_index):(
                acc_index+MemoryExtension.width-MemoryTable.width)]
            acc_index += MemoryExtension.width-MemoryTable.width
            if len(input_symbols) != 0:
                input_point += tuples[index][(acc_index):(
                    acc_index+IOExtension.width-IOTable.width)]
                acc_index += IOExtension.width-IOTable.width
            if len(output_symbols) != 0:
                output_point += tuples[index][(acc_index):(
                    acc_index+IOExtension.width-IOTable.width)]
                acc_index += IOExtension.width-IOTable.width
            assert(acc_index == len(
                tuples[index]), "Column count in verifier must match until end")

            # ******************** processor quotients ********************
            # boundary
            for constraint, bound in zip(processor_extension.boundary_constraints_ext(), processor_extension.boundary_quotient_degree_bounds(log_time)):
                eval = constraint.evaluate(processor_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after processor boundaries: ", len(terms))

            # transition
            unit_distance = processor_extension.unit_distance(
                fri.domain.length)
            next_index = (index + unit_distance) % fri.domain.length
            next_processor_point = tuples[next_index][num_randomizer_polynomials:(
                num_randomizer_polynomials+ProcessorTable.width)]
            next_processor_point += tuples[next_index][extension_offset:(
                extension_offset+ProcessorExtension.width-ProcessorTable.width)]
            challenges = [a, b, c, d, e, f, alpha, beta, gamma, delta]
            for constraint, bound in zip(processor_extension.transition_constraints_ext(challenges), processor_extension.transition_quotient_degree_bounds(log_time, challenges)):
                eval = constraint.evaluate(
                    processor_point + next_processor_point)
                quotient = eval * self.xfield.lift(fri.domain(index) - processor_extension.omicron.inverse()) / (
                    self.xfield.lift(fri.domain(index) ^ BrainfuckStark.roundup_npo2(processor_extension.get_height())) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after processor transitions: ", len(terms))

            # terminal
            challenges = [a, b, c, d, e, f, alpha, beta, gamma, delta]
            terminals = [processor_instruction_permutation_terminal, processor_memory_permutation_terminal,
                         processor_input_evaluation_terminal, processor_output_evaluation_terminal]
            for constraint, bound in zip(processor_extension.terminal_constraints_ext(challenges, terminals), processor_extension.terminal_quotient_degree_bounds(log_time, challenges, terminals)):
                eval = constraint.evaluate(processor_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) -
                     self.xfield.lift(processor_extension.omicron.inverse()))
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after processor terminals: ", len(terms))

            # ******************** instruction quotients ********************
            # boundary
            for constraint, bound in zip(instruction_extension.boundary_constraints_ext(), instruction_extension.boundary_quotient_degree_bounds(log_instructions)):
                eval = constraint.evaluate(instruction_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after instruction boundaries: ", len(terms))

            # transition
            unit_distance = instruction_extension.unit_distance(
                fri.domain.length)
            next_index = (index + unit_distance) % fri.domain.length
            next_instruction_point = tuples[next_index][(num_randomizer_polynomials+ProcessorTable.width):(
                num_randomizer_polynomials+ProcessorTable.width+InstructionTable.width)]
            next_instruction_point += tuples[next_index][(extension_offset+ProcessorExtension.width-ProcessorTable.width):(
                extension_offset+ProcessorExtension.width-ProcessorTable.width+InstructionExtension.width-InstructionTable.width)]
            challenges = [a, b, c, alpha, eta]
            for constraint, bound in zip(instruction_extension.transition_constraints_ext(challenges), instruction_extension.transition_quotient_degree_bounds(log_instructions, challenges)):
                eval = constraint.evaluate(
                    instruction_point + next_instruction_point)
                quotient = eval * self.xfield.lift(fri.domain(index) - instruction_extension.omicron.inverse()) / (
                    self.xfield.lift(fri.domain(index) ^ BrainfuckStark.roundup_npo2(instruction_extension.get_height())) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after instruction transitions: ", len(terms))

            # terminal
            challenges = [a, b, c, alpha, eta]
            terminals = [processor_instruction_permutation_terminal,
                         instruction_evaluation_terminal]
            for constraint, bound in zip(instruction_extension.terminal_constraints_ext(challenges, terminals), instruction_extension.terminal_quotient_degree_bounds(log_instructions, challenges, terminals)):
                eval = constraint.evaluate(instruction_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) -
                     self.xfield.lift(instruction_extension.omicron.inverse()))
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after instruction terminals: ", len(terms))

            # ******************** memory quotients ********************
            # boundary
            for constraint, bound in zip(memory_extension.boundary_constraints_ext(), memory_extension.boundary_quotient_degree_bounds(log_time)):
                eval = constraint.evaluate(memory_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]

            # transition
            unit_distance = memory_extension.unit_distance(fri.domain.length)
            next_index = (index + unit_distance) % fri.domain.length
            next_memory_point = tuples[next_index][(num_randomizer_polynomials+ProcessorTable.width+InstructionTable.width):(
                num_randomizer_polynomials+ProcessorTable.width+InstructionTable.width+MemoryTable.width)]
            next_memory_point += tuples[next_index][(extension_offset+ProcessorExtension.width-ProcessorTable.width+InstructionExtension.width-InstructionTable.width):(
                extension_offset+ProcessorExtension.width-ProcessorTable.width+InstructionExtension.width-InstructionTable.width+MemoryExtension.width-MemoryTable.width)]
            challenges = [d, e, f, beta]
            for constraint, bound in zip(memory_extension.transition_constraints_ext(challenges), memory_extension.transition_quotient_degree_bounds(log_time, challenges)):
                eval = constraint.evaluate(memory_point + next_memory_point)
                quotient = eval * self.xfield.lift(fri.domain(index) - memory_extension.omicron.inverse()) / (
                    self.xfield.lift(fri.domain(index) ^ BrainfuckStark.roundup_npo2(memory_extension.get_height())) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]

            # terminal
            challenges = [d, e, f, beta]
            terminals = [processor_memory_permutation_terminal]
            for constraint, bound in zip(memory_extension.terminal_constraints_ext(challenges, terminals), memory_extension.terminal_quotient_degree_bounds(log_time, challenges, terminals)):
                eval = constraint.evaluate(memory_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) -
                     self.xfield.lift(memory_extension.omicron.inverse()))
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after memory: ", len(terms))

            # ******************** input quotients ********************
            # boundary
            for constraint, bound in zip(input_extension.boundary_constraints_ext(), input_extension.boundary_quotient_degree_bounds(log_input)):
                eval = constraint.evaluate(input_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]

            # transition
            if input_extension.get_height() > 0:
                unit_distance = input_extension.unit_distance(
                    fri.domain.length)
                next_index = (index + unit_distance) % fri.domain.length
                next_input_point = tuples[next_index][(num_randomizer_polynomials+ProcessorTable.width+InstructionTable.width+MemoryTable.width):(
                    num_randomizer_polynomials+ProcessorTable.width+InstructionTable.width+MemoryTable.width+IOTable.width)]
                next_input_point += tuples[next_index][(extension_offset+ProcessorExtension.width-ProcessorTable.width+InstructionExtension.width-InstructionTable.width+MemoryExtension.width-MemoryTable.width):(
                    extension_offset+ProcessorExtension.width-ProcessorTable.width+InstructionExtension.width-InstructionTable.width+MemoryExtension.width-MemoryTable.width+IOExtension.width-IOTable.width)]
                challenges = [gamma]
                for constraint, bound in zip(input_extension.transition_constraints_ext(challenges), input_extension.transition_quotient_degree_bounds(log_input, challenges)):
                    eval = constraint.evaluate(input_point + next_input_point)
                    quotient = eval * self.xfield.lift(fri.domain(index) - input_extension.omicron.inverse()) / (
                        self.xfield.lift(fri.domain(index) ^ BrainfuckStark.roundup_npo2(input_extension.get_height())) - self.xfield.one())
                    terms += [quotient]
                    shift = max_degree - bound
                    terms += [quotient *
                              self.xfield.lift(fri.domain(index) ^ shift)]

            # terminal
            challenges = [gamma]
            terminals = [processor_input_evaluation_terminal]
            for constraint, bound in zip(input_extension.terminal_constraints_ext(challenges, terminals), input_extension.terminal_quotient_degree_bounds(log_input, challenges, terminals)):
                eval = constraint.evaluate(input_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) -
                     self.xfield.lift(input_extension.omicron.inverse()))
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after input: ", len(terms))

            # ******************** output quotients ********************
            # boundary
            for constraint, bound in zip(output_extension.boundary_constraints_ext(), output_extension.boundary_quotient_degree_bounds(log_output)):
                eval = constraint.evaluate(output_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) - self.xfield.one())
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]

            # transition
            if output_extension.get_height() > 0:
                unit_distance = output_extension.unit_distance(
                    fri.domain.length)
                next_index = (index + unit_distance) % fri.domain.length
                next_output_point = tuples[next_index][(num_randomizer_polynomials+ProcessorTable.width+InstructionTable.width+MemoryTable.width+IOTable.width):(
                    num_randomizer_polynomials+ProcessorTable.width+InstructionTable.width+MemoryTable.width+IOTable.width+IOTable.width)]
                next_output_point += tuples[next_index][(extension_offset+ProcessorExtension.width-ProcessorTable.width+InstructionExtension.width-InstructionTable.width+MemoryExtension.width-MemoryTable.width+IOExtension.width-IOTable.width):(
                    extension_offset+ProcessorExtension.width-ProcessorTable.width+InstructionExtension.width-InstructionTable.width+MemoryExtension.width-MemoryTable.width+IOExtension.width-IOTable.width+IOExtension.width-IOTable.width)]
                challenges = [delta]
                for constraint, bound in zip(output_extension.transition_constraints_ext(challenges), output_extension.transition_quotient_degree_bounds(log_output, challenges)):
                    eval = constraint.evaluate(
                        output_point + next_output_point)
                    quotient = eval * self.xfield.lift(fri.domain(index) - output_extension.omicron.inverse()) / (
                        self.xfield.lift(fri.domain(index) ^ BrainfuckStark.roundup_npo2(output_extension.get_height())) - self.xfield.one())
                    terms += [quotient]
                    shift = max_degree - bound
                    terms += [quotient *
                              self.xfield.lift(fri.domain(index) ^ shift)]

            # terminal
            challenges = [delta]
            terminals = [processor_output_evaluation_terminal]
            for constraint, bound in zip(output_extension.terminal_constraints_ext(challenges, terminals), output_extension.terminal_quotient_degree_bounds(log_output, challenges, terminals)):
                eval = constraint.evaluate(output_point)
                quotient = eval / \
                    (self.xfield.lift(fri.domain(index)) -
                     self.xfield.lift(output_extension.omicron.inverse()))
                terms += [quotient]
                shift = max_degree - bound
                terms += [quotient *
                          self.xfield.lift(fri.domain(index) ^ shift)]
            print("len(terms) after output: ", len(terms))

            # differences
            # quotient_codewords += [[(processor_codewords[ProcessorExtension.instruction_permutation][i] -
            #                      instruction_codewords[InstructionExtension.permutation][i]) * self.xfield.lift((fri.domain(i) - self.field.one()).inverse()) for i in range(fri.domain.length)]]
            # quotient_codewords += [[(processor_codewords[ProcessorExtension.memory_permutation][i] -
            #                     memory_codewords[MemoryExtension.permutation][i]) * self.xfield.lift((fri.domain(i) - self.field.one()).inverse()) for i in range(fri.domain.length)]]
            # quotient_degree_bounds += [(1 << log_instructions) -
            #                        2, (1 << log_time) - 2]
            difference = (processor_point[ProcessorExtension.instruction_permutation] -
                          instruction_point[InstructionExtension.permutation])
            quotient = difference / \
                (self.xfield.lift(fri.domain(index)) - self.xfield.one())
            terms += [quotient]
            shift = max_degree - ((1 << log_instructions) - 2)
            terms += [quotient * self.xfield.lift(fri.domain(index) ^ shift)]

            difference = (
                processor_point[ProcessorExtension.memory_permutation] - memory_point[MemoryExtension.permutation])
            quotient = difference / \
                (self.xfield.lift(fri.domain(index)) - self.xfield.one())
            terms += [quotient]
            shift = max_degree - ((1 << log_time) - 2)
            terms += [quotient * self.xfield.lift(fri.domain(index) ^ shift)]

            assert(len(terms) == len(
                weights)), f"length of terms ({len(terms)}) must be equal to length of weights ({len(weights)})"

            print("verifier terms:")
            for t in terms:
                print(t)

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
        print("starting FRI verification ...")
        tick = time.time()
        polynomial_points = []
        verifier_verdict = fri.verify(proof_stream, combination_root)
        polynomial_points.sort(key=lambda iv: iv[0])
        if verifier_verdict == False:
            return False
        tock = time.time()
        print("FRI verification took", (tock - tick), "seconds")

        # deprecate: use FRI-indices to verify nonlicombo

        # indices = [i for i, _ in polynomial_points]
        # values = [v for _, v in polynomial_points]

        # # process indices
        # duplicated_indices = [i for i in indices] + \
        #     [(i + self.expansion_factor) %
        #      fri.domain.length for i in indices]
        # quadrupled_indices = [i for i in duplicated_indices] + [
        #     (i + (fri.domain.length // 2)) % fri.domain.length for i in duplicated_indices]
        # quadrupled_indices.sort()

        # # get leafs
        # print("number of quadrupled indices:", len(quadrupled_indices))
        # base_leafs = []
        # extension_leafs = []
        # for i in quadrupled_indices:
        #     base_leafs += [proof_stream.pull()]
        #     print("<- base leaf", i)
        #     extension_leafs += [proof_stream.pull()]
        #     print("<- extension leaf", i)

        # # get authentication paths
        # base_paths = []
        # extension_paths = []
        # for i in quadrupled_indices:
        #     base_paths += [proof_stream.pull()]
        #     print("<- base path", i)
        #     extension_paths += [proof_stream.pull()]
        #     print("<- extension path", i)

        # # verify authentication paths
        # for qi, (elm, salt), path in zip(quadrupled_indices, base_leafs, base_paths):
        #     SaltedMerkle.verify(base_root, qi, salt, path, elm)
        # for qi, (elm, salt), path in zip(quadrupled_indices, extension_leafs, extension_paths):
        #     SaltedMerkle.verify(base_root, qi, salt, path, elm)

        # # compute extension degree bounds
        # extension_degree_bounds = []
        # extension_degree_bounds += [processor_extension.get_height()-1] * (
        #     ProcessorExtension.width - ProcessorTable.width)
        # extension_degree_bounds += [instruction_extension.get_height()-1] * (
        #     InstructionExtension.width - InstructionTable.width)
        # extension_degree_bounds += [memory_extension.get_height()-1] * \
        #     (MemoryExtension.width - MemoryTable.width)
        # if input_extension.get_height() != 0:
        #     extension_degree_bounds += [input_extension.get_height()-1] * \
        #         (IOExtension.width - IOTable.width)
        # if output_extension.get_height() != 0:
        #     extension_degree_bounds += [output_extension.get_height()-1] * \
        #         (IOExtension.width - IOTable.width)

        # # compute quotient degree bounds
        # quotient_degree_bounds = []
        # print("number of degree bounds:")
        # quotient_degree_bounds += processor_extension.all_quotient_degree_bounds(log_time, challenges=[a, b, c, d, e, f, alpha, beta, gamma, delta], terminals=[
        #     processor_instruction_permutation_terminal, processor_memory_permutation_terminal, processor_input_evaluation_terminal, processor_output_evaluation_terminal])
        # print(len(quotient_degree_bounds))
        # quotient_degree_bounds += instruction_extension.all_quotient_degree_bounds(log_time, challenges=[a, b, c, alpha, eta], terminals=[
        #     processor_instruction_permutation_terminal, instruction_evaluation_terminal])
        # print(len(quotient_degree_bounds))
        # quotient_degree_bounds += memory_extension.all_quotient_degree_bounds(log_time, challenges=[
        #     d, e, f, beta], terminals=[processor_memory_permutation_terminal])
        # print(len(quotient_degree_bounds))
        # quotient_degree_bounds += input_extension.all_quotient_degree_bounds(
        #     log_input, challenges=[gamma], terminals=[processor_input_evaluation_terminal])
        # print(len(quotient_degree_bounds))
        # quotient_degree_bounds += output_extension.all_quotient_degree_bounds(
        #     log_output, challenges=[delta], terminals=[processor_output_evaluation_terminal])
        # print(len(quotient_degree_bounds))
        # quotient_degree_bounds += [(1 << log_instructions) -
        #                            2, (1 << log_time) - 2]

        # # verify nonlinear combination
        # for index, y in polynomial_points:

        #     sum = self.xfield.zero()
        #     for j in range(num_base_polynomials):
        #         shiftj = max_degree - base_degree_bounds[j]
        #         sum += weights[2*j] * base_leafs[0][j] + \
        #             weights[2*j+1] * base_leafs[0][j] * (omega ^ shiftj)
        #     for j in range(num_randomizer_polynomials):
        #         sum += weights[2*num_base_polynomials+j] * \
        #             base_leafs[0][2*num_base_polynomials+j]
        #     for j in range(num_extension_polynomials):
        #         shiftj = max_degree - extension_degree_bounds[j]
        #         sum += weights[2*num_base_polynomials + num_randomizer_polynomials + 2*j] * extension_leafs[0][j] + \
        #             weights[2*num_base_polynomials + num_randomizer_polynomials +
        #                     2*j+1] * extension_leafs[0][j] * (omega ^ shiftj)
        #     for j in range(len(quotient_degree_bounds)):
        #         shiftj = max_degree - quotient_degree_bounds[j]
        #         sum += weights[2*num_base_polynomials + num_randomizer_polynomials + 2*num_extension_polynomials + 2*j] * extension_leafs[0][num_extension_polynomials+j] + \
        #             weights[2*num_base_polynomials + num_randomizer_polynomials + 2 *
        #                     num_extension_polynomials + 2*j + 1] * extension_leafs[0][num_extension_polynomials+j] * (omega ^ shiftj)

        #     verifier_verdict = verifier_verdict and (y == sum)
        #     if not verifier_verdict:
        #         return False

        # # verify air constraints
        # for i in range(len(quadrupled_indices)-1):
        #     qi, (base_elm, _), (sec_elm, _) = zip(
        #         quadrupled_indices, base_leafs, extension_leafs)[i]
        #     qi_next, (base_elm_next, _), (sec_elm_next, _) = zip(
        #         quadrupled_indices, base_leafs, extension_leafs)[i+1]

        #     if qi_next == qi + 1:
        #         current_index = qi

        #         point = base_elm + \
        #             sec_elm[0:num_extension_polynomials]
        #         quotients_from_leafs = sec_elm[num_extension_polynomials:]
        #         shifted_point = base_elm_next + \
        #             sec_elm_next[0:num_extension_polynomials]

        #         # internal airs
        #         evaluated_quotients = []
        #         evaluated_quotients += [processor_extension.evaluate_quotients(
        #             omicron, omega ^ current_index, point, shifted_point)]
        #         evaluated_quotients += [instruction_extension.evaluate_quotients(
        #             omicron, omega ^ current_index, point, shifted_point)]
        #         evaluated_quotients += [memory_extension.evaluate_quotients(
        #             omicron, omega ^ current_index, point, shifted_point)]
        #         evaluated_quotients += [input_extension.evaluate_quotients(
        #             omicron, omega ^ current_index, point, shifted_point)]
        #         evaluated_quotients += [output_extension.evaluate_quotients(
        #             omicron, omega ^ current_index, point, shifted_point)]

        #         # table relations
        #         # X = Polynomial([self.xfield.zero(), self.xfield.one()])
        #         # quotient_polynomials += [(processor_extension_polynomials[ProcessorExtension.instruction_permutation] -
        #         #                         instruction_extension_polynomials[InstructionExtension.permutation]) / (X - self.xfield.one())]
        #         # quotient_polynomials += [(processor_extension_polynomials[ProcessorExtension.memory_permutation] -
        #         #                         memory_extension_polynomials[MemoryExtension.permutation]) / (X - self.xfield.one())]
        #         evaluated_quotients += [(sec_elm[ProcessorExtension.instruction_permutation - ProcessorTable().width] - sec_elm
        #                                  [processor_extension.width - ProcessorTable().width + InstructionExtension.permutation - InstructionTable().width]) / ((omega ^ current_index) - self.xfield.one())]
        #         evaluated_quotients += [(sec_elm[ProcessorExtension.memory_permutation - ProcessorTable().width] - sec_elm
        #                                  [processor_extension.width - ProcessorTable().width + instruction_extension.width - InstructionTable().width + MemoryExtension.permutation - MemoryTable().width]) / ((omega ^ current_index) - self.xfield.one())]

        #         verifier_verdict = verifier_verdict and evaluated_quotients == quotients_from_leafs

        # verify external terminals:
        # input
        verifier_verdict = verifier_verdict and processor_extension.input_evaluation_terminal == VirtualMachine.evaluation_terminal(
            [self.xfield.lift(t) for t in input_symbols], gamma)
        # output
        verifier_verdict = verifier_verdict and processor_extension.input_evaluation_terminal == VirtualMachine.evaluation_terminal(
            [self.xfield.lift(t) for t in output_symbols], delta)
        # program
        print(type(instruction_evaluation_terminal))
        print(type(VirtualMachine.program_evaluation(
            program, a, b, c, eta)))
        verifier_verdict = verifier_verdict and instruction_evaluation_terminal == VirtualMachine.program_evaluation(
            program, a, b, c, eta)

        return verifier_verdict
