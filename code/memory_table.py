from aet import *
from processor_table import ProcessorTable


class MemoryTable(Table):
    # name columns
    cycle = 0
    memory_pointer = 1
    memory_value = 2

    width = 3

    def __init__(self, field):
        super(MemoryTable, self).__init__(field, 3)

    def pad(self, padded_processor_table):
        current_cycle = max(row[MemoryTable.cycle].value for row in self.table)
        while len(self.table) & (len(self.table)-1):
            current_cycle += 1
            new_row = [self.field.zero()] * self.width
            new_row[MemoryTable.cycle] = padded_processor_table.table[current_cycle][ProcessorTable.cycle]
            new_row[MemoryTable.memory_pointer] = padded_processor_table.table[current_cycle][ProcessorTable.memory_pointer]
            new_row[MemoryTable.memory_value] = padded_processor_table.table[current_cycle][ProcessorTable.memory_value]
            self.table += [new_row]

        self.table.sort(key=lambda r: r[1].value)

    @staticmethod
    def transition_constraints_afo_named_variables(cycle, address, value, cycle_next, address_next, value_next):
        field = list(address.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())

        polynomials = []

        # 1. memory pointer increases by one, zero, or minus one
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
            cycle_next, address_next, value_next = MPolynomial.variables(
                6, self.field)
        return MemoryTable.transition_constraints_afo_named_variables(cycle, address, value, cycle_next, address_next, value_next)

    def boundary_constraints(self):
        # format: mpolynomial
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [x[MemoryTable.cycle],
                x[MemoryTable.memory_pointer],
                x[MemoryTable.memory_value],
                ]
