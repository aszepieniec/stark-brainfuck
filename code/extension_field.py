from univariate import *
from algebra import *


class ExtensionFieldElement:
    def __init__(self, polynomial, field):
        self.polynomial = Polynomial(
            polynomial.coefficients[:polynomial.degree()+1])
        self.field = field

    def __add__(self, right):
        return self.field.add(self, right)

    def __mul__(self, right):
        return self.field.multiply(self, right)

    def __sub__(self, right):
        return self.field.subtract(self, right)

    def __truediv__(self, right):
        return self.field.divide(self, right)

    def __neg__(self):
        return self.field.negate(self)

    def inverse(self):
        return self.field.inverse(self)

    # modular exponentiation -- be sure to encapsulate in parentheses!
    def __xor__(self, exponent):
        acc = self.field.one()
        val = ExtensionFieldElement(self.polynomial, self.field)
        for i in reversed(range(len(bin(exponent)[2:]))):
            acc = acc * acc
            if (1 << i) & exponent != 0:
                acc = acc * val
        return acc

    def __eq__(self, other):
        return self.polynomial == other.polynomial

    def __neq__(self, other):
        return self.polynomial != other.polynomial

    def __str__(self):
        return str(self.polynomial)

    def __bytes__(self):
        return bytes("|".join(str(c.polynomial) for c in self.polynomial.coefficients).encode())

    def is_zero(self):
        return self.polynomial.is_zero()


class ExtensionField:
    def __init__(self, modulus):
        self.modulus = modulus

    def zero(self):
        return ExtensionFieldElement(Polynomial([]), self)

    def one(self):
        return ExtensionFieldElement(Polynomial([self.modulus.coefficients[0].field.one()]), self)

    def multiply(self, left, right):
        return ExtensionFieldElement((left.polynomial * right.polynomial) % self.modulus, self)

    def add(self, left, right):
        return ExtensionFieldElement(left.polynomial + right.polynomial, self)

    def subtract(self, left, right):
        return ExtensionFieldElement(left.polynomial - right.polynomial, self)

    def negate(self, operand):
        return ExtensionFieldElement(-operand.polynomial, self)

    def inverse(self, operand):
        a, b, g = Polynomial.xgcd(operand.polynomial, self.modulus)
        assert(a * operand.polynomial + b *
               self.modulus == g), "bezout relation fails"
        return ExtensionFieldElement(a % self.modulus, self)

    def divide(self, left, right):
        assert(not right.is_zero()), "divide by zero"
        a, b, g = Polynomial.xgcd(right.polynomial, self.modulus)
        return ExtensionFieldElement(left.polynomial * a % self.modulus, self)

    def main():
        # p = 2^64 - 2^32 + 1
        #   = 1 + 3 * 5 * 17 * 257 * 65537 * 2^32
        #   = 1 + 4294967295 * 2^32
        p = 18446744069414584321  # 2^64 - 2^32 + 1
        field = BaseField(p)
        one = BaseFieldElement(1, field)
        minus_one = BaseFieldElement(p-1, field)
        # modulus = X^3 - X + 1
        modulus = Polynomial([one, minus_one, field.zero(), one])
        return ExtensionField(modulus)

    def sample(self, byte_array):
        chunk_length = len(byte_array) // self.modulus.degree()
        start = 0
        stop = chunk_length
        coefficients = []
        for i in range(self.modulus.degree()):
            element = self.modulus.coefficients[0].field.sample(
                byte_array[start:stop])
            start += chunk_length
            stop += chunk_length
            coefficients += [element]
        return ExtensionFieldElement(Polynomial(coefficients), self)

    def lift(self, base_field_element: BaseFieldElement) -> ExtensionFieldElement:
        if type(base_field_element) == ExtensionFieldElement:
            return base_field_element
        return ExtensionFieldElement(Polynomial([base_field_element]), self)

    def __str__(self):
        return self.polynomial.__str__()

    def __call__(self, integer):
        return ExtensionFieldElement(Polynomial([BaseFieldElement(integer, self.modulus.coefficients[0].field)]), self)
