from univariate import *


class MPolynomial:
    def __init__(self, dictionary):
        # Multivariate polynomials are represented as dictionaries with exponent vectors
        # as keys and coefficients as values. E.g.:
        # f(x,y,z) = 17 + 2xy + 42z - 19x^6*y^3*z^12 is represented as:
        # {
        #     (0,0,0) => 17,
        #     (1,1,0) => 2,
        #     (0,0,1) => 42,
        #     (6,3,12) => -19,
        # }
        self.dictionary = dictionary

    def zero():
        return MPolynomial(dict())

    def __add__(self, other):
        dictionary = dict()
        num_variables = max([0] + [len(k) for k in self.dictionary.keys()
                                   ] + [len(k) for k in other.dictionary.keys()])
        for k, v in self.dictionary.items():
            pad = list(k) + [0] * (num_variables - len(k))
            pad = tuple(pad)
            dictionary[pad] = v
        for k, v in other.dictionary.items():
            pad = list(k) + [0] * (num_variables - len(k))
            pad = tuple(pad)
            if pad in dictionary.keys():
                dictionary[pad] = dictionary[pad] + v
            else:
                dictionary[pad] = v
        return MPolynomial(dictionary)

    def __mul__(self, other):
        dictionary = dict()
        num_variables = max([len(k) for k in self.dictionary.keys(
        )] + [len(k) for k in other.dictionary.keys()])
        for k0, v0 in self.dictionary.items():
            for k1, v1 in other.dictionary.items():
                exponent = [0] * num_variables
                for k in range(len(k0)):
                    exponent[k] += k0[k]
                for k in range(len(k1)):
                    exponent[k] += k1[k]
                exponent = tuple(exponent)
                if exponent in dictionary.keys():
                    dictionary[exponent] = dictionary[exponent] + v0 * v1
                else:
                    dictionary[exponent] = v0 * v1
        return MPolynomial(dictionary)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        dictionary = dict()
        for k, v in self.dictionary.items():
            dictionary[k] = -v
        return MPolynomial(dictionary)

    def __xor__(self, exponent):
        if self.is_zero():
            return MPolynomial(dict())
        field = list(self.dictionary.values())[0].field
        num_variables = len(list(self.dictionary.keys())[0])
        exp = [0] * num_variables
        acc = MPolynomial({tuple(exp): field.one()})
        for b in bin(exponent)[2:]:
            acc = acc * acc
            if b == '1':
                acc = acc * self
        return acc

    def constant(element):
        return MPolynomial({tuple([0]): element})

    def is_zero(self):
        if not self.dictionary:
            return True
        else:
            for v in self.dictionary.values():
                if v.is_zero() == False:
                    return False
            return True

    def degree(self):
        if not self.dictionary:
            return -1
        return max(sum(k) for k in self.dictionary.keys())

    # Returns the multivariate polynomials representing each indeterminates linear function
    # with a leading coefficient of one. For three indeterminates, returns:
    # [f(x,y,z) = x, f(x,y,z) = y, f(x,y,z) = z]
    def variables(num_variables: int, field):
        variables = []
        for i in range(num_variables):
            exponent = [0] * i + [1] + [0] * (num_variables - i - 1)
            variables = variables + \
                [MPolynomial({tuple(exponent): field.one()})]
        return variables

    def evaluate(self, point):
        acc = point[0].field.zero()
        if self.degree() == -1:
            return acc
        for k, v in self.dictionary.items():
            prod = v
            assert(len(point) == len(
                k)), f"number of elements in point {len(point)} does not match with number of variables {len(k)} for polynomial {str(self)}"
            for i in range(len(k)):
                prod = prod * (point[i] ^ k[i])
            acc = acc + prod
        return acc

    def evaluate_symbolic(self, point, memo=dict()):
        field = list(self.dictionary.values())[0].field
        acc = Polynomial([])
        for k, v in self.dictionary.items():
            prod = Polynomial([field.one()])
            for i in range(len(k)):
                inneracc = Polynomial([field.one()])
                j = 0
                if (i, 1 << j) not in memo:
                    memo[(i, 1 << j)] = point[i]
                while (1 << j) <= k[i]:
                    if (i, 1 << j) not in memo:
                        pointij = memo.get((i, 1 << (j-1)))
                        pointij = pointij * pointij
                        memo[(i, 1 << j)] = pointij
                    else:
                        pointij = memo[(i, 1 << j)]
                    if (k[i] & (1 << j)) != 0:
                        inneracc *= pointij
                    j += 1
                prod *= inneracc
            acc += prod * Polynomial([v])
        return acc

    def symbolic_degree_bound(self, max_degrees):
        """Given a vector `max_degrees` of degree bounds on the arguments,
        compute the smallest degree bound on the univariate polynomial
        resulting from symbolic evaluation in a vector of polynomials
        satisfying the degree bounds `max_degrees`.
        Specifically, if `self.evaluate_symbolic` computes the map
            (f_0(x), f_1(x), f_2(x)) ---> u(x)
        and
            forall i . degree(f_i(x)) <= `max_degrees[i]`
        then
            degree(u(x)) <= `self.symbolic_degree(max_degrees)`
        """
        if self.degree() == -1:
            return -1
        total_degree_bound = -1
        assert(len(max_degrees) >= len(list(self.dictionary.keys())[
               0])), f"max degrees length ({len(max_degrees)}) does not match with number of variables (key for first term: {list(self.dictionary.keys())[0]})"
        assert(max_degrees == [max_degrees[0]] * len(max_degrees)
               ), "max degrees must be n repetitions of the same integer"
        for exponents, coefficient in self.dictionary.items():
            if coefficient.is_zero():
                continue
            term_degree_bound = 0
            for e, md in zip(exponents, max_degrees):
                term_degree_bound += e * md
            total_degree_bound = max(total_degree_bound, term_degree_bound)
        return total_degree_bound

    def lift(polynomial, variable_index):
        if polynomial.is_zero():
            return MPolynomial({})
        field = polynomial.coefficients[0].field
        variables = MPolynomial.variables(variable_index+1, field)
        x = variables[-1]
        acc = MPolynomial({})
        for i in range(len(polynomial.coefficients)):
            acc = acc + \
                MPolynomial.constant(polynomial.coefficients[i]) * (x ^ i)
        return acc

    def __str__(self):
        return " + ".join(str(value) + "*" + "*".join("x" + str(i) + "^" + str(key[i]) for i in range(len(key)) if key[i] != 0) for key, value in self.dictionary.items())

    def partial_evaluate(self, partial_assignment):
        field = list(self.dictionary.values())[0].field
        num_variables = len(list(self.dictionary.keys())[0])
        variables = MPolynomial.variables(num_variables, field)

        complete_assignment = variables
        for key, value in partial_assignment.items():
            complete_assignment[key] = MPolynomial.constant(value)

        polynomial = MPolynomial.zero()
        for key, value in self.dictionary.items():
            term = MPolynomial.constant(value)
            for i in range(num_variables):
                term *= complete_assignment[i] ^ key[i]
            polynomial += term

        return polynomial
