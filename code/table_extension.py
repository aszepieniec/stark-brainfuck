from abc import abstractmethod
from aet import Table
from ntt import batch_inverse
from processor_table import ProcessorTable
from univariate import Polynomial


class TableExtension(Table):
    def __init__(self, xfield, original_width, width):
        super().__init__(xfield, width)
        self.original_width = original_width
        self.xfield = xfield

    def interpolate_extension(self, omega, order, num_randomizers):
        return self.interpolate_columns(omega, order, num_randomizers, range(self.original_width, self.width))

    @abstractmethod
    def boundary_constraints_ext(self):
        pass

    def boundary_quotients(self, omicron, fri_domain, codewords):
        quotient_codewords = []
        boundary_constraints = self.boundary_constraints_ext()
        print("got", len(boundary_constraints), "boundary constraints")
        zerofier = [fri_domain(i) - omicron.field.one() for i in range(fri_domain.length)]
        zerofier_inverse = batch_inverse(zerofier)

        for l in range(len(boundary_constraints)):
            mpo = boundary_constraints[l]
            quotient_codewords += [[mpo.evaluate([codewords[j][i] for j in range(
                self.width)]) * self.xfield.lift(zerofier_inverse[i]) for i in range(fri_domain.length)]]
        return quotient_codewords

    def boundary_quotient_degree_bounds(self, log_num_rows):
        if log_num_rows >= 0:
            composition_degree = (1 << log_num_rows) - 1
        else:
            composition_degree = -1
        return [composition_degree - 1] * len(self.boundary_constraints_ext())

    @abstractmethod
    def transition_constraints_ext(self, challenges):
        pass

    def transition_quotients(self, omicron, log_num_rows, domain, codewords, challenges):
        if log_num_rows < 0:
            return []
            
        interpolation_subgroup_order = 1 << log_num_rows
        print("interpolation subgroup order:", interpolation_subgroup_order)
        quotients = []
        field = domain.omega.field
        subgroup_zerofier = [(domain(i) ^ interpolation_subgroup_order) - field.one() for i in range(domain.length)]
        subgroup_zerofier_inverse = batch_inverse(subgroup_zerofier)
        zerofier_inverse = [subgroup_zerofier_inverse[i] * (domain(i) - omicron.inverse()) for i in range(domain.length)]

        transition_constraints = self.transition_constraints_ext(challenges)
        print("got", len(transition_constraints), "transition constraints")

        symbolic_point = [domain.xinterpolate(c) for c in codewords]
        symbolic_point = symbolic_point + [sp.scale(self.xfield.lift(omicron)) for sp in symbolic_point]
        X = Polynomial([self.field.zero(), self.field.one()])
        symbolic_zerofier = (((X^interpolation_subgroup_order)) - Polynomial([self.field.one()])) / (X - Polynomial([self.field.lift(omicron.inverse())]))

        # for i in range(interpolation_subgroup_order):
        #     print("value of symbolic zerofier in omicron^%i:" % i, symbolic_zerofier.evaluate(self.field.lift(omicron^i)))

        for l in range(len(transition_constraints)):
            mpo = transition_constraints[l]
            quotient_codeword = []
            for i in range(domain.length):
                point = [codewords[j][i] for j in range(self.width)] + \
                         [codewords[j][(i+(domain.length // interpolation_subgroup_order)) % domain.length] for j in range(self.width)]
                quotient_codeword += [mpo.evaluate(point) * self.field.lift(zerofier_inverse[i])]
            
            quotients += [quotient_codeword]

            if l == -1:
                print("symbolically evaluating polynomial", mpo)
                symbolic_transition_polynomial = mpo.evaluate_symbolic(symbolic_point)
                print("transition quotient degree:", domain.xinterpolate(quotients[-1]).degree(), "versus codeword length:", len(quotients[-1]))
                print("symbolic transition polynomial degree:", symbolic_transition_polynomial.degree())
                for i in range(interpolation_subgroup_order):
                    print("value in omicron^%i:" % i, symbolic_transition_polynomial.evaluate(self.field.lift(omicron^i)))
                symbolic_quotient, symbolic_remainder = symbolic_transition_polynomial.divide(symbolic_zerofier)
                print("symbolic quotient degree:", symbolic_quotient.degree())
                print("symbolic remainder degree:", symbolic_remainder.degree())
                print("---")
        return quotients

    def transition_quotient_degree_bounds(self, log_num_rows, challenges):
        if log_num_rows >= 0:
            trace_degree = (1 << log_num_rows)-1
        else:
            trace_degree = -1
        air_degree = max(air.degree()
                         for air in self.transition_constraints_ext(challenges))
        composition_degree = trace_degree * air_degree
        return [composition_degree - trace_degree] * len(self.transition_constraints_ext(challenges))

    @ abstractmethod
    def terminal_constraints_ext(self, challenges, terminals):
        pass

    def terminal_quotients(self, omicron, domain, codewords, challenges, terminals):
        quotient_codewords = []
        zerofier = [domain(i) - omicron.inverse()
                    for i in range(domain.length)]
        zerofier_inverse = batch_inverse(zerofier)
        for mpo in self.terminal_constraints_ext(challenges, terminals):
            quotient_codewords += [[mpo.evaluate([codewords[j][i] for j in range(
                self.width)]) * self.field.lift(zerofier_inverse[i]) for i in range(domain.length)]]
        return quotient_codewords

    def terminal_quotient_degree_bounds(self, log_num_rows, challenges, terminals):
        if log_num_rows >= 0:
            degree = (1 << log_num_rows) - 1
        else:
            degree = -1
        air_degree = max(tc.degree()
                         for tc in self.terminal_constraints_ext(challenges, terminals))
        return [air_degree * degree - 1] * len(self.terminal_constraints_ext(challenges, terminals))

    def all_quotients(self, omicron, domain, codewords, log_num_rows, challenges, terminals):
        boundary_quotients = self.boundary_quotients(omicron, domain, codewords)
        transition_quotients = self.transition_quotients(omicron, log_num_rows, domain, codewords, challenges)
        terminal_quotients = self.terminal_quotients(omicron, domain, codewords, challenges, terminals)
        print("have quotients:", len(boundary_quotients), "boundary,", len(transition_quotients), "transition, and", len(terminal_quotients), "terminal")
        return boundary_quotients + transition_quotients + terminal_quotients

    def all_quotient_degree_bounds(self, log_num_rows, challenges, terminals):
        bounds = self.boundary_quotient_degree_bounds(log_num_rows) + self.transition_quotient_degree_bounds(
            log_num_rows, challenges) + self.terminal_quotient_degree_bounds(log_num_rows, challenges, terminals)
        return bounds

    def num_quotients(self):
        return len(self.all_quotient_degree_bounds())

    def evaluate_boundary_quotients(self, omicron, omegai, point):
        values = []
        for cycle, mpo in self.boundary_constraints_ext():
            values += mpo.evaluate(point) / (omegai - (omicron ^ cycle))
        return values

    def evaluate_transition_quotients(self, omicron, omegai, point, shifted_point, log_num_rows, challenges):
        values = []
        zerofier = (omegai ^ (1 << log_num_rows) - 1) / \
            (omegai - omicron.inverse())
        for mpo in self.transition_constraints_ext(challenges):
            values += [mpo.evaluate(point + shifted_point) / zerofier]
        return values

    def evaluate_terminal_quotients(self, omicron, omegai, point, shifted_point, challenges, terminals):
        values = []
        zerofier = omegai - omicron.inverse()
        for mpo in self.terminal_constraints_ext(challenges, terminals):
            values += [mpo.evaluate(point+shifted_point) / zerofier]
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
