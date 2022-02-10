from abc import abstractmethod
from table import Table
from ntt import batch_inverse
from os import environ
from processor_table import ProcessorTable
from univariate import Polynomial


class TableExtension(Table):
    def __init__(self, xfield, base_width, width, height, generator, order):
        super().__init__(xfield, width, height, generator, order)
        self.base_width = base_width
        self.xfield = xfield

    def interpolate_extension(self, omega, order, num_randomizers):
        print("hi from table_extension")
        return self.interpolate_columns(omega, order, num_randomizers, range(self.base_width, self.width))

    @abstractmethod
    def boundary_constraints_ext(self):
        pass

    def boundary_quotients(self, fri_domain, codewords):
        assert(len(codewords) !=
               0), "'codewords' argument must have nonzero length"

        quotient_codewords = []
        boundary_constraints = self.boundary_constraints_ext()
        zerofier = [fri_domain(i) - fri_domain.omega.field.one()
                    for i in range(fri_domain.length)]
        zerofier_inverse = batch_inverse(zerofier)

        for l in range(len(boundary_constraints)):
            mpo = boundary_constraints[l]
            quotient_codewords += [[mpo.evaluate([codewords[j][i] for j in range(
                self.width)]) * self.xfield.lift(zerofier_inverse[i]) for i in range(fri_domain.length)]]

        if environ.get('DEBUG') is not None:
            print(f"before domain interpolation of bq in {type(self)}")
            for qc in quotient_codewords:
                interpolated = fri_domain.xinterpolate(qc)
                print(f"degree of interpolation: {interpolated.degree()}")
                assert(interpolated.degree() < fri_domain.length - 1)
            print("Done!")

        return quotient_codewords

    def boundary_quotient_degree_bounds(self):
        composition_degree = self.get_interpolant_degree() - 1
        return [composition_degree - 1] * len(self.boundary_constraints_ext())

    @abstractmethod
    def transition_constraints_ext(self, challenges):
        pass

    def transition_quotients(self, domain, codewords, challenges):

        # TODO: include number of randomizers in calculation
        interpolation_subgroup_order = self.get_interpolation_domain_length()
        quotients = []
        field = domain.omega.field
        subgroup_zerofier = [(domain(
            i) ^ interpolation_subgroup_order) - field.one() for i in range(domain.length)]
        subgroup_zerofier_inverse = batch_inverse(subgroup_zerofier)
        zerofier_inverse = [subgroup_zerofier_inverse[i] *
                            (domain(i) - self.omicron.inverse()) for i in range(domain.length)]

        transition_constraints = self.transition_constraints_ext(challenges)

        # symbolic_point = [domain.xinterpolate(c) for c in codewords]
        # symbolic_point = symbolic_point + \
        #     [sp.scale(self.xfield.lift(self.omicron)) for sp in symbolic_point]
        # X = Polynomial([self.field.zero(), self.field.one()])
        # symbolic_zerofier = (((X ^ interpolation_subgroup_order)) - Polynomial(
        #     [self.field.one()])) / (X - Polynomial([self.field.lift(self.omicron.inverse())]))

        # for i in range(interpolation_subgroup_order):
        #     print("value of symbolic zerofier in omicron^%i:" % i, symbolic_zerofier.evaluate(self.field.lift(omicron^i)))

        for l in range(len(transition_constraints)):
            mpo = transition_constraints[l]
            quotient_codeword = []
            for i in range(domain.length):
                point = [codewords[j][i] for j in range(self.width)] + \
                    [codewords[j][(i+self.unit_distance(domain.length)) %
                                  domain.length] for j in range(self.width)]
                quotient_codeword += [mpo.evaluate(point)
                                      * self.field.lift(zerofier_inverse[i])]

            quotients += [quotient_codeword]

            if environ.get('DEBUG') is not None:
                print(f"before domain interpolation of tq in {type(self)}")
                interpolated = domain.xinterpolate(quotients[-1])
                print(f"degree of interpolation: {interpolated.degree()}")
                assert(interpolated.degree() < domain.length - 1)
                assert(domain.xinterpolate(
                    quotients[-1]).degree() < domain.length-1), f"quotient polynomial has maximal degree in table {type(self)}"
                print("Done!")

        return quotients

    def transition_quotient_degree_bounds(self, challenges):
        trace_degree = self.get_interpolant_degree()
        air_degree = max(air.degree()
                         for air in self.transition_constraints_ext(challenges))
        composition_degree = trace_degree * air_degree
        return [composition_degree - trace_degree] * len(self.transition_constraints_ext(challenges))

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
                self.width)]) * self.field.lift(zerofier_inverse[i]) for i in range(domain.length)]]

        if environ.get('DEBUG') is not None:
            print(
                f"before domain interpolation of term quotients in {type(self)}")
            for qc in quotient_codewords:
                interpolated = domain.xinterpolate(qc)
                print(f"degree of interpolation: {interpolated.degree()}")
                assert(interpolated.degree() < domain.length - 1)
            print("Done!")

        return quotient_codewords

    def terminal_quotient_degree_bounds(self, challenges, terminals):
        degree = self.get_interpolant_degree()
        air_degree = max(tc.degree()
                         for tc in self.terminal_constraints_ext(challenges, terminals))
        return [air_degree * degree - 1] * len(self.terminal_constraints_ext(challenges, terminals))

    def all_quotients(self, domain, codewords, challenges, terminals):
        boundary_quotients = self.boundary_quotients(
            domain, codewords)
        transition_quotients = self.transition_quotients(
            domain, codewords, challenges)
        terminal_quotients = self.terminal_quotients(
            domain, codewords, challenges, terminals)
        return boundary_quotients + transition_quotients + terminal_quotients

    def all_quotient_degree_bounds(self, challenges, terminals):
        bounds = self.boundary_quotient_degree_bounds() + self.transition_quotient_degree_bounds(
            challenges) + self.terminal_quotient_degree_bounds(challenges, terminals)
        return bounds

    def num_quotients(self):
        return len(self.all_quotient_degree_bounds(self.challenges, self.terminals))

    def evaluate_boundary_quotients(self, omicron, omegai, point):
        values = []
        for cycle, mpo in self.boundary_constraints_ext():
            values += mpo.evaluate(point) / (omegai - (omicron ^ cycle))
        return values

    def evaluate_transition_quotients(self, omicron, omegai, point, next_point, challenges):
        values = []
        zerofier = (omegai ^ Table.roundup_npo2(self.get_height()) - 1) / \
            (omegai - omicron.inverse())
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
            + self.evaluate_transition_quotients(omicron,
                                                 omegai, point, shifted_point, self.log_num_rows, self.challenges) \
            + self.evaluate_terminal_quotients(omicron,
                                               point, self.log_num_rows, self.challenges, self.terminals)

    @ staticmethod
    def prepare_verify(log_num_rows, challenges, terminals):
        pass

    def test(self):
        for i in range(len(self.boundary_constraints_ext())):
            mpo = self.boundary_constraints_ext()[i]
            if len(self.table) != 0:
                point = self.table[0]
                assert(mpo.evaluate(point).is_zero(
                )), f"BOUNDARY constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"

        transition_constraints = self.transition_constraints_ext(
            self.challenges)
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

        terminal_constraints = self.terminal_constraints_ext(
            self.challenges, self.terminals)
        if len(self.table) != 0:
            for i in range(len(terminal_constraints)):
                mpo = terminal_constraints[i]
                point = self.table[-1]
                assert(mpo.evaluate(point).is_zero(
                )), f"TERMINAL constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"
