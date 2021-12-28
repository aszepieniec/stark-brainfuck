from aet import *

class MemoryTable(Table):
    def __init__(self, field):
        super(MemoryTable, self).__init__(field, 3)

    @staticmethod
    def transition_constraints_afo_named_variables( cycle, address, value, cycle_next, address_next, value_next ):
        field = list(address.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())

        polynomials = []

        # 1. memory pointer increases by one or zero
        # <=>. (MP*=MP+1) \/ (MP*=MP)
        polynomials += [(address_next - address - one)
                 * (address_next - address)]

        # 2. if memory pointer does not increase, then memory value can change only if cycle counter increases by one
        #        <=>. MP*=MP => (MV*=/=MV => CLK*=CLK+1)
        #        <=>. MP*=/=MP \/ (MV*=/=MV => CLK*=CLK+1)
        # (DNF:) <=>. MP*=/=MP \/ MV*=MV \/ CLK*=CLK+1
        polynomials += [(address_next - address - one) *
                 (value_next - value) * (cycle_next - cycle - one)]

        # 3. if memory pointer increases by one, then memory value must be set to zero
        #        <=>. MP*=MP+1 => MV* = 0
        # (DNF:) <=>. MP*=/=MP+1 \/ MV*=0
        polynomials += [(address_next - address)
                 * value_next]
        
        return polynomials

    def transition_constraints(self):
        cycle, address, value, \
            cycle_next, address_next, value_next = MPolynomial.variables(6, self.field)
        return MemoryTable.transition_constraints_afo_named_variables(cycle, address, value, cycle_next, address_next, value_next)

    def boundary_constraints(self):
        return [(0, 0, self.field.zero()),  # cycle
                (1, 0, self.field.zero()),  # memory pointer
                (2, 0, self.field.zero()),  # memory value
                ]