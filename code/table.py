from abc import abstractmethod
from random import random
from multivariate import *
from ntt import *
import os


class Table:
    def __init__(self, field, base_width, full_width, length, num_randomizers, generator, order):
        self.field = field
        self.base_width = base_width
        self.full_width = full_width
        self.length = length
        self.num_randomizers = num_randomizers
        self.height = Table.roundup_npo2(length)
        self.omicron = Table.derive_omicron(
            generator, order, self.height)
        self.generator = generator
        self.order = order
        self.matrix = []

    @staticmethod
    def roundup_npo2(integer):
        if integer == 0:
            return 0
        elif integer == 1:
            return 1
        return 1 << (len(bin(integer-1)[2:]))

    @staticmethod
    def derive_omicron(generator, generator_order, target_order):
        while generator_order != target_order:
            generator = generator ^ 2
            generator_order = generator_order // 2
        return generator

    def unit_distance(self, omega_order):
        if self.height == 0:
            return 0
        return omega_order // self.height

    def get_interpolation_domain_length(self):
        return self.height + self.num_randomizers

    def interpolant_degree(self):
        return self.get_interpolation_domain_length() - 1

    def test(self):
        for i in range(len(self.base_boundary_constraints())):
            mpo = self.base_boundary_constraints()[i]
            if len(self.matrix) != 0:
                point = self.matrix[0]
                assert(mpo.evaluate(point).is_zero(
                )), f"BOUNDARY constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"

        transition_constraints = self.base_transition_constraints()
        for i in range(len(transition_constraints)):
            mpo = transition_constraints[i]
            for rowidx in range(len(self.matrix)-1):
                assert(len(self.matrix[rowidx]) == len(
                    self.matrix[rowidx+1])), "table has consecutive rows of different length"
                point = self.matrix[rowidx] + self.matrix[rowidx+1]
                assert(len(point) == len(list(mpo.dictionary.keys())[
                       0])), f"point has {len(point)} elements but mpo has {len(list(mpo.dictionary.keys())[0])} variables .."
                assert(mpo.evaluate(point).is_zero(
                )), f"TRNASITION constraint {i} not satisfied in row {rowidx}; point: {[str(p) for p in point]}; polynomial {str(mpo.partial_evaluate({1: point[1]}))} evaluates to {str(mpo.evaluate(point))}"

    def xtest(self, challenges, terminals):
        # test boundary constraints
        boundary_constraints = self.boundary_constraints_ext(challenges)
        for i in range(len(boundary_constraints)):
            if self.length == 0:
                continue
            mpo = boundary_constraints[i]
            point = [self.matrix[0][j] for j in range(self.full_width)]
            assert(mpo.evaluate(point).is_zero())

        # test transition constraints
        transition_constraints = self.transition_constraints_ext(challenges)
        for i in range(len(transition_constraints)):
            if self.length == 0:
                continue
            mpo = transition_constraints[i]
            for j in range(self.height-1):
                point = [self.matrix[j][k] for k in range(
                    self.full_width)] + [self.matrix[j+1][k] for k in range(self.full_width)]
                assert(mpo.evaluate(point).is_zero())

        # test terminal constraints
        terminal_constraints = self.terminal_constraints_ext(
            challenges, terminals)
        for i in range(len(terminal_constraints)):
            if self.length == 0:
                continue
            mpo = terminal_constraints[i]
            point = [self.matrix[self.height-1][j]
                     for j in range(self.full_width)]
            if not mpo.evaluate(point).is_zero():
                print("self.height:", self.height)
                print("self.length:", self.length)
                print("constraint evaluated:", mpo.evaluate(point))
                print("evaluation column in last row:",
                      self.matrix[self.height-1][1])
                print("evaluation column in row index length-1:",
                      self.matrix[self.length-1][1])
                print("offset should be:",
                      self.matrix[self.height-1][1] / self.matrix[self.length-1][1])
                print("terminal index:", self.terminal_index)
                print("indicated challenge:", challenges[self.terminal_index])
                assert(False)

    def interpolate_columns(self, omega, omega_order, column_indices):
        assert(omega.has_order_po2(omega_order)
               ), "omega does not have claimed order"

        if self.height == 0:
            return [Polynomial([])] * len(column_indices)

        polynomials = []
        omicron_domain = [self.field.lift(self.omicron ^ i)
                          for i in range(self.height)]
        randomizer_domain = [self.field.lift(omega^(2*i+1)) # odd powers of omega => no collision with omicron
                             for i in range(self.num_randomizers)]
        domain = omicron_domain + randomizer_domain
        for c in column_indices:
            trace = [row[c] for row in self.matrix]
            randomizers = [self.field.sample(os.urandom(3*8))
                           for i in range(self.num_randomizers)]
            values = trace + randomizers
            assert(len(values) == len(
                domain)), f"length of domain {len(domain)} and values {len(values)} are unequal"
            polynomials += [fast_interpolate(domain,
                                             values,
                                             self.field.lift(omega), omega_order)]

        return polynomials

    def lde(self, domain):
        polynomials = self.interpolate_columns(
            domain.omega, domain.length, column_indices=range(self.base_width))
        self.codewords = [domain.evaluate(p) for p in polynomials]
        return self.codewords

    def ldex(self, domain, xfield):
        polynomials = self.interpolate_columns(
            domain.omega, domain.length, column_indices=range(self.base_width, self.full_width))
        codewords = [domain.xevaluate(p, xfield) for p in polynomials]
        self.codewords += codewords
        return codewords

    @abstractmethod
    def boundary_constraints_ext(self):
        pass

    def boundary_quotients(self, fri_domain, codewords, challenges):
        assert(len(codewords) !=
               0), "'codewords' argument must have nonzero length"

        quotient_codewords = []
        boundary_constraints = self.boundary_constraints_ext(challenges)
        zerofier = [fri_domain(i) - fri_domain.omega.field.one()
                    for i in range(fri_domain.length)]
        zerofier_inverse = batch_inverse(zerofier)

        for l in range(len(boundary_constraints)):
            mpo = boundary_constraints[l]
            quotient_codewords += [[mpo.evaluate([codewords[j][i] for j in range(
                self.full_width)]) * self.field.lift(zerofier_inverse[i]) for i in range(fri_domain.length)]]

        if os.environ.get('DEBUG') is not None:
            print(f"before domain interpolation of bq in {type(self)}")
            for qc in quotient_codewords:
                interpolated = fri_domain.xinterpolate(qc)
                print(f"degree of interpolation: {interpolated.degree()}")
                assert(interpolated.degree() < fri_domain.length - 1)
            print("Done!")

        return quotient_codewords

    def boundary_quotient_degree_bounds(self, challenges):
        max_degrees = [self.interpolant_degree()] * self.full_width
        degree_bounds = [mpo.symbolic_degree_bound(
            max_degrees) - 1 for mpo in self.boundary_constraints_ext(challenges)]
        return degree_bounds

    @abstractmethod
    def transition_constraints_ext(self, challenges):
        pass

    def transition_quotients(self, domain, codewords, challenges):

        quotients = []
        field = domain.omega.field
        subgroup_zerofier = [(domain(i) ^ self.height) - field.one()
                             for i in range(domain.length)]
        if self.height != 0:
            subgroup_zerofier_inverse = batch_inverse(subgroup_zerofier)
        else:
            subgroup_zerofier_inverse = subgroup_zerofier
        zerofier_inverse = [subgroup_zerofier_inverse[i] *
                            (domain(i) - self.omicron.inverse()) for i in range(domain.length)]

        transition_constraints = self.transition_constraints_ext(challenges)

        for l in range(len(transition_constraints)):
            mpo = transition_constraints[l]
            quotient_codeword = []
            composition_codeword = []
            for i in range(domain.length):
                point = [codewords[j][i] for j in range(self.full_width)] + \
                    [codewords[j][(i+self.unit_distance(domain.length)) %
                                  domain.length] for j in range(self.full_width)]
                composition_codeword += [mpo.evaluate(point)]
                quotient_codeword += [mpo.evaluate(point)
                                      * self.field.lift(zerofier_inverse[i])]

            quotients += [quotient_codeword]

            if os.environ.get('DEBUG') is not None:
                print(f"before domain interpolation of tq in {type(self)}")
                interpolated = domain.xinterpolate(quotients[-1])
                print(f"degree of interpolation: {interpolated.degree()}")
                if interpolated.degree() >= domain.length - 1:
                    print("terminal index:", self.terminal_index)
                    print("self.height:", self.height)
                    print("point:", ",".join(str(p) for p in point))
                    print("codeword:", ",".join(str(c)
                          for c in composition_codeword[:5]))
                    print("quotient:", ",".join(str(c)
                          for c in quotient_codeword[:5]))
                    assert(False)
                assert(domain.xinterpolate(
                    quotients[-1]).degree() < domain.length-1), f"quotient polynomial has maximal degree in table {type(self)}"
                print("Done!")

        return quotients

    def transition_quotient_degree_bounds(self, challenges):
        max_degrees = [self.interpolant_degree()] * (2*self.full_width)

        degree_bounds = []
        transition_constraints = self.transition_constraints_ext(challenges)
        for i in range(len(transition_constraints)):
            mpo = transition_constraints[i]
            symbolic_degree_bound = mpo.symbolic_degree_bound(max_degrees)
            degree_bounds += [symbolic_degree_bound - self.height + 1]
        return degree_bounds

    @abstractmethod
    def terminal_constraints_ext(self, challenges, terminals):
        pass

    def terminal_quotients(self, domain, codewords, challenges, terminals):
        quotient_codewords = []

        zerofier_codeword = [domain(i) - self.omicron.inverse()
                             for i in range(domain.length)]

        zerofier_inverse = batch_inverse(zerofier_codeword)
        for mpo in self.terminal_constraints_ext(challenges, terminals):
            quotient_codewords += [[mpo.evaluate([codewords[j][i] for j in range(
                self.full_width)]) * self.field.lift(zerofier_inverse[i]) for i in range(domain.length)]]

        if os.environ.get('DEBUG') is not None:
            for i in range(len(quotient_codewords)):
                qc = quotient_codewords[i]
                interpolated = domain.xinterpolate(qc)
                if interpolated.degree() >= domain.length - 1:
                    print("interpolated degree:", interpolated.degree())
                    print("domain length:", domain.length)
                    print("interpolation degree failure")
                    constraint = self.terminal_constraints_ext(
                        challenges, terminals)[i]
                    codeword = [constraint.evaluate([codewords[j][i] for j in range(
                        self.full_width)]) for i in range(domain.length)]
                    interpolated = domain.xinterpolate(codeword)
                    reevaluated = [interpolated.evaluate(self.field.lift(
                        self.omicron ^ i)) for i in range(self.height)]
                    print("reevaluated:", ",".join(str(c)
                          for c in reevaluated))
                    print(type(interpolated))
                    print(type(codeword))
                    print(type(domain))
                    assert(False)

        return quotient_codewords

    def terminal_quotient_degree_bounds(self, challenges, terminals):
        max_degrees = [self.interpolant_degree()] * self.full_width
        degree_bounds = [mpo.symbolic_degree_bound(
            max_degrees) - 1 for mpo in self.terminal_constraints_ext(challenges, terminals)]
        return degree_bounds

    def all_quotients(self, domain, codewords, challenges, terminals):
        boundary_quotients = self.boundary_quotients(
            domain, codewords, challenges)
        transition_quotients = self.transition_quotients(
            domain, codewords, challenges)
        terminal_quotients = self.terminal_quotients(
            domain, codewords, challenges, terminals)
        return boundary_quotients + transition_quotients + terminal_quotients

    def all_quotient_degree_bounds(self, challenges, terminals):
        boundary_degree_bounds = self.boundary_quotient_degree_bounds(
            challenges)
        transition_degree_bounds = self.transition_quotient_degree_bounds(
            challenges)
        terminal_degree_bounds = self.terminal_quotient_degree_bounds(
            challenges, terminals)
        return boundary_degree_bounds + transition_degree_bounds + terminal_degree_bounds

    def num_quotients(self, challenges, terminals):
        return len(self.all_quotient_degree_bounds(challenges, terminals))

    def evaluate_boundary_quotients(self, omicron, omegai, point):
        values = []
        for cycle, mpo in self.boundary_constraints_ext():
            values += mpo.evaluate(point) / (omegai - (omicron ^ cycle))
        return values

    def evaluate_transition_quotients(self, omicron, omegai, point, next_point, challenges):
        values = []
        zerofier = (omegai ^ Table.roundup_npo2(
            self.get_height()) - 1) / (omegai - omicron.inverse())
        for mpo in self.transition_constraints_ext(challenges):
            values += [mpo.evaluate(point + next_point) / zerofier]
        return values

    def evaluate_terminal_quotients(self, omicron, omegai, point, next_point, challenges, terminals):
        values = []
        zerofier = omegai - omicron.inverse()
        for mpo in self.terminal_constraints_ext(challenges, terminals):
            values += [mpo.evaluate(point+next_point) / zerofier]
        return values

    def evaluate_quotients(self, omicron, omegai, point, shifted_point):
        return self.evaluate_boundary_quotients(omicron, omegai, point) \
            + self.evaluate_transition_quotients(
            omicron, omegai, point, shifted_point, self.log_num_rows, self.challenges) \
            + self.evaluate_terminal_quotients(omicron, point,
                                               self.log_num_rows, self.challenges, self.terminals)
