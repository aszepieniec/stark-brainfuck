
from ntt import batch_inverse


class PermutationArgument:
    def __init__(self, lhs_table, lhs_column_index, rhs_table, rhs_column_index):
        self.lhs_table = lhs_table
        self.lhs_column_index = lhs_column_index
        self.rhs_table = rhs_table
        self.rhs_column_index = rhs_column_index

    def quotient(self, fri_domain):
        field = fri_domain.omega.field
        difference_codeword = [l - r for l, r in zip(self.lhs_table.codewords[self.lhs_column_index],
                                                     self.rhs_table.codewords[self.rhs_column_index])]
        zerofier = [fri_domain(i) - field.one()
                    for i in range(fri_domain.length)]
        zerofier_inverse = batch_inverse(zerofier)
        quotient_codeword = [d * d.field.lift(z)
                             for d, z in zip(difference_codeword, zerofier_inverse)]
        return quotient_codeword

    def quotient_degree_bound(self):
        lhs_interpolant_degree = self.lhs_table.interpolant_degree()
        rhs_interpolant_degree = self.rhs_table.interpolant_degree()
        # print("interpolant degrees ---\n rhs:",
        #       rhs_interpolant_degree, "\nlhs:", lhs_interpolant_degree)
        degree = max(self.lhs_table.interpolant_degree(),
                     self.rhs_table.interpolant_degree())
        return degree - 1
