
from ntt import batch_inverse


class PermutationArgument:
    def __init__(self, all_tables, lhs, rhs):
        self.all_tables = all_tables
        self.lhs = lhs
        self.rhs = rhs

    def quotient(self, fri_domain):
        field = fri_domain.omega.field
        print("len of self.all_tables:", len(self.all_tables))
        print("self.lhs:", self.lhs)
        print("self.rhs:", self.rhs)
        print("width of lhs table:", self.all_tables[self.lhs[0]].width)
        print("width of rhs table:", self.all_tables[self.rhs[0]].width)
        print("number of lhs codewords:", len(self.all_tables[self.lhs[0]].codewords))
        print("number of rhs codewords:", len(self.all_tables[self.rhs[0]].codewords))
        difference_codeword = [l - r for l, r in zip(self.all_tables[self.lhs[0]].codewords[self.lhs[1]],
                                                     self.all_tables[self.rhs[0]].codewords[self.rhs[1]])]
        zerofier = [fri_domain(i) - field.one()
                    for i in range(fri_domain.length)]
        zerofier_inverse = batch_inverse(zerofier)
        quotient_codeword = [d * d.field.lift(z)
                             for d, z in zip(difference_codeword, zerofier_inverse)]
        return quotient_codeword

    def quotient_degree_bound(self):
        lhs_interpolant_degree = self.all_tables[self.lhs[0]].interpolant_degree()
        rhs_interpolant_degree = self.all_tables[self.rhs[0]].interpolant_degree()
        # print("interpolant degrees ---\n rhs:",
        #       rhs_interpolant_degree, "\nlhs:", lhs_interpolant_degree)
        degree = max(lhs_interpolant_degree,
                     rhs_interpolant_degree)
        return degree - 1
