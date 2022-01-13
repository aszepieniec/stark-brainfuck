from abc import abstractmethod
from aet import Table
from univariate import Polynomial

class TableExtension(Table):
    def __init__( self, xfield, original_width, width ):
        super().__init__(xfield, width)
        self.original_width = original_width
        self.xfield = xfield

    def interpolate_extension( self, omega, order, num_randomizers ):
        return self.interpolate_columns(omega, order, num_randomizers, range(self.original_width, self.width))

    @abstractmethod
    def boundary_constraints_ext(self):
        pass

    def boundary_quotients(self):
        quotients = []
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        for row, mpo in self.boundary_constraints():
            composition_polynomial = mpo.symbolic_evaluate(self.polynomials)
            quotients += composition_polynomial / (X - self.xfield(row))
        return quotients

    @staticmethod
    def boundary_quotient_degree_bounds(log_num_rows):
        composition_degree = (1 << log_num_rows) - 1
        return composition_degree - 1

    @abstractmethod
    def transition_constraints_ext( self, challenges ):
        pass

    def transition_quotients( self, challenges ):
        quotients = []
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        for mpo in self.transition_constraints(challenges):
            point = self.polynomials + [p.scale(self.omicron) for p in self.polynomials]
            composition_polynomial = mpo.symbolic_evaluate(point)
            quotient = composition_polynomial * (X - self.omicron.inverse()) / (X^self.domain_length - 1)
            quotients += [quotient]
        return quotients

    def transition_quotient_degree_bounds(self, log_num_rows, challenges):
        air_degree = max(air.degree() for air in self.transition_constraints(challenges))
        composition_degree = ((1 << log_num_rows) - 1) * air_degree
        return composition_degree + 1 - (1 << log_num_rows)

    @abstractmethod
    def terminal_constraints_ext(self, challenges, terminals):
        pass

    def terminal_quotients( self, challenges, terminals ):
        quotients = []
        X = Polynomial([self.xfield.zero(), self.xfield.one()])
        for mpo in self.terminal_constraints(challenges, terminals):
            quotient = mpo.symbolic_evaluate(self.polynomials) / (X - Polynomial([self.omicron.inverse()]))
            quotients += [quotient]
        return quotients
    
    def terminal_quotient_degree_bounds(self, log_num_rows, challenges, terminals):
        degree = (1 << log_num_rows) - 1
        air_degree = max(tc.degree() for tc in self.terminal_constraints(challenges, terminals))
        return air_degree * degree - 1

    def test(self):
        for i in range(len(self.boundary_constraints_ext())):
            row, mpo = self.boundary_constraints_ext()[i]
            if len(self.table) != 0:
                point = self.table[row]
                assert(mpo.evaluate(point).is_zero(
                )), f"BOUNDARY constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"

        transition_constraints = self.transition_constraints_ext(self.challenges)
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

        terminal_constraints = self.terminal_constraints_ext(self.challenges, self.terminals)
        if len(self.table) != 0:
            for i in range(len(terminal_constraints)):
                mpo = terminal_constraints[i]
                point = self.table[-1]
                assert(mpo.evaluate(point).is_zero()), f"TERMINAL constraint {i} not satisfied; point: {[str(p) for p in point]}; polynomial {str(mpo)} evaluates to {str(mpo.evaluate(point))}"