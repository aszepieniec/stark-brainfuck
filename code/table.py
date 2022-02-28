from abc import abstractmethod
from random import random
from multivariate import *
from ntt import *
import os


class Table:
    def __init__(self, field, width, length, num_randomizers, generator, order):
        self.field = field
        self.width = width
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
        return omega_order // Table.roundup_npo2(self.height)

    def get_interpolation_domain_length(self):
        return Table.roundup_npo2(self.height) + self.num_randomizers

    def get_interpolant_degree(self):
        return self.get_interpolation_domain_length() - 1

    # @abstractmethod
    # @staticmethod
    # def transition_constraints():
    #     pass

    # @abstractmethod
    # @staticmethod
    # def boundary_constraints():
    #     pass

    def test(self):
        for i in range(len(self.boundary_constraints())):
            mpo = self.boundary_constraints()[i]
            if len(self.matrix) != 0:
                point = self.matrix[0]
                assert(mpo.evaluate(point).is_zero(
                )), f"BOUNDARY constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"

        transition_constraints = self.transition_constraints()
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

    def interpolate(self, omega, order):
        return self.interpolate_columns(omega, order, column_indices=range(self.width))

    def interpolate_columns(self, omega, omega_order, column_indices):
        assert(omega.has_order_po2(omega_order)
               ), "omega does not have claimed order"
        print("called interpolate_columns with omega:", omega, "order:", omega_order,
              "num randomizers:", self.num_randomizers, "table height:", len(self.matrix))

        if self.height == 0:
            return [Polynomial([])] * len(column_indices)

        polynomials = []
        omicron_domain = [self.field.lift(self.omicron ^ i)
                          for i in range(self.height)]
        print("length of omicron domain:", len(omicron_domain))
        randomizer_domain = [self.field.lift(omega) * omicron_domain[i]
                             for i in range(self.num_randomizers)]
        print("length of randomizer domain:", len(randomizer_domain))
        domain = omicron_domain + randomizer_domain
        for c in column_indices:
            trace = [row[c] for row in self.matrix]
            randomizers = [self.field.sample(os.urandom(3*8))
                           for i in range(self.num_randomizers)]
            values = trace + randomizers
            # assert(len(values) == len(
            #     domain)), f"length of domain {len(domain)} and values {len(values)} are unequal"
            polynomials += [fast_interpolate(domain,
                                             values,
                                             self.field.lift(omega), omega_order)]

        return polynomials
