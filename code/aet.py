from abc import abstractmethod
from random import random
from multivariate import *
from ntt import *
import os


class Table:
    def __init__(self, field, width, height, generator, order):
        self.field = field
        self.width = width
        self.height = height
        self.omicron = Table.derive_omicron(
            generator, order, Table.roundup_npo2(height))
        self.generator = generator
        self.order = order
        self.table = []

    @staticmethod
    def roundup_npo2(integer):
        if integer == 0 or integer == 1:
            return 1
        return 1 << (len(bin(integer-1)[2:]))

    @staticmethod
    def derive_omicron(generator, generator_order, target_order):
        while generator_order != target_order:
            generator = generator ^ 2
            generator_order = generator_order // 2
        return generator

    def unit_distance(self, omega_order):
        return omega_order // Table.roundup_npo2(self.get_height())

    def nrows(self):
        return len(self.table)

    def ncols(self):
        return self.width

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
            if len(self.table) != 0:
                point = self.table[0]
                assert(mpo.evaluate(point).is_zero(
                )), f"BOUNDARY constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"

        transition_constraints = self.transition_constraints()
        for i in range(len(transition_constraints)):
            mpo = transition_constraints[i]
            for rowidx in range(self.nrows()-1):
                assert(len(self.table[rowidx]) == len(
                    self.table[rowidx+1])), "table has consecutive rows of different length"
                point = self.table[rowidx] + self.table[rowidx+1]
                assert(len(point) == len(list(mpo.dictionary.keys())[
                       0])), f"point has {len(point)} elements but mpo has {len(list(mpo.dictionary.keys())[0])} variables .."
                assert(mpo.evaluate(point).is_zero(
                )), f"TRNASITION constraint {i} not satisfied in row {rowidx}; point: {[str(p) for p in point]}; polynomial {str(mpo.partial_evaluate({1: point[1]}))} evaluates to {str(mpo.evaluate(point))}"

    def interpolate(self, omega, order, num_randomizers):
        return self.interpolate_columns(omega, order, num_randomizers, column_indices=range(self.width))

    def interpolate_columns(self, omega, omega_order, num_randomizers, column_indices):
        print("called interpolate_columns with omega:", omega, "order:", omega_order,
              "num randomizers:", num_randomizers, "table length:", len(self.table))

        num_rows = len(self.table)
        if num_rows == 0:
            return []

        assert(num_rows != 0), "number of rows cannot be zero"

        assert(num_rows & (num_rows - 1) ==
               0), f"num_rows has value {num_rows} but must be a power of two"

        assert(omega ^ omega_order == omega.field.one()
               ), "order must match with omega"
        assert(omega ^ (omega_order//2) != omega.field.one()
               ), "order must be primitive"

        omidi_order = 1
        while omidi_order < num_rows + num_randomizers:
            omidi_order = omidi_order << 1

        randomness_expansion_factor = int(omidi_order / num_rows)

        omidi = omega
        current_order = omega_order
        while current_order != omidi_order:
            omidi = omidi ^ 2
            current_order = current_order // 2

        assert(omidi ^ omidi_order == omidi.field.one()
               ), "order must now be order of omidi"
        assert(omidi ^ (omidi_order//2) !=
               omidi.field.one()), "order not primitive"

        omicron = omidi
        omicron_order = omidi_order
        while omicron_order != num_rows:
            omicron = omicron ^ 2
            omicron_order = omicron_order // 2

        self.omidi = omidi
        self.omicron = omicron

        assert(omidi.has_order_po2(omidi_order)
               ), "omidi does not have right order"
        assert(omicron.has_order_po2(num_rows)
               ), "omicron does not have right order"

        polynomials = []
        for col in column_indices:
            trace = [self.field.sample(os.urandom(3*8))
                     for row in range(omidi_order)]
            trace = [self.field.zero()
                     for row in range(omidi_order)]
            for row in range(num_rows):
                trace[randomness_expansion_factor*row] = self.table[row][col]
            polynomials += [Polynomial(intt(self.field.lift(omidi), trace))]

            for row in range(num_rows):
                assert(
                    polynomials[-1].evaluate(self.field.lift(omicron ^ row)) == self.table[row][col])

        return polynomials

    def get_height(self):
        if self.table:
            return len(self.table)
        elif hasattr(self, 'height'):
            return self.height
        else:
            return 0
