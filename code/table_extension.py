from abc import abstractmethod
from aet import Table
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

    def boundary_quotients(self):
        quotients = []
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        for row, mpo in self.boundary_constraints_ext():
            composition_polynomial = mpo.evaluate_symbolic(self.polynomials)
            quotients += [composition_polynomial / (X - Polynomial([self.omicron^row]))]
        return quotients

    @staticmethod
    def boundary_quotient_degree_bounds(log_num_rows):
        composition_degree = (1 << log_num_rows) - 1
        return composition_degree - 1

    @abstractmethod
    def transition_constraints_ext(self, challenges):
        pass

    def transition_quotients(self, challenges):
        quotients = []
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        one = Polynomial([self.xfield.one()])
        for mpo in self.transition_constraints_ext(challenges):
            point = self.polynomials + \
                [p.scale(self.omicron) for p in self.polynomials]
            composition_polynomial = mpo.evaluate_symbolic(point)
            quotient = composition_polynomial * \
                (X - self.omicron.inverse()) / ((X ^ self.domain_length) - one)
            quotients += [quotient]
        return quotients

    def transition_quotient_degree_bounds(self, log_num_rows, challenges):
        air_degree = max(air.degree()
                         for air in self.transition_constraints_ext(challenges))
        composition_degree = ((1 << log_num_rows) - 1) * air_degree
        return composition_degree + 1 - (1 << log_num_rows)

    @abstractmethod
    def terminal_constraints_ext(self, challenges, terminals):
        pass

    def terminal_quotients(self, challenges, terminals):
        quotients = []
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        for mpo in self.terminal_constraints_ext(challenges, terminals):
            quotient = mpo.evaluate_symbolic(
                self.polynomials) / (X - Polynomial([self.omicron.inverse()]))
            quotients += [quotient]
        return quotients

    def terminal_quotient_degree_bounds(self, log_num_rows, challenges, terminals):
        degree = (1 << log_num_rows) - 1
        air_degree = max(tc.degree()
                         for tc in self.terminal_constraints())
        return air_degree * degree - 1

    def all_quotients(self, log_num_rows, challenges, terminals):
        return self.boundary_quotients() + self.transition_quotients(challenges) + self.terminal_quotients(challenges, terminals)

    def all_quotient_degree_bounds(self):
        return self.boundary_quotient_degree_bounds() + self.transition_quotient_degree_bounds() + self.terminal_quotient_degree_bounds()

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
            + self.evaluate_transition_quotients(omicron, omegai, point, shifted_point, self.log_num_rows, self.challenges) \
            + self.evaluate_terminal_quotients(omicron,
                                               point, self.log_num_rows, self.challenges, self.terminals)

    @staticmethod
    def prepare_verify(log_num_rows, challenges, terminals):
        pass

    def test(self):
        for i in range(len(self.boundary_constraints_ext())):
            row, mpo = self.boundary_constraints_ext()[i]
            if len(self.table) != 0:
                point = self.table[row]
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
