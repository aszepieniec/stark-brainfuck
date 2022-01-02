from io_table import *

class IOExtension(IOTable):
    def __init__(self, gamma):
        field = gamma.field

        # names for challenges
        self.gamma = MPolynomial.constant(gamma)

        super(IOExtension, self).__init__(field)
        self.width = 1+2

    @staticmethod
    def extend(input_table, gamma):

        # algebra stuff
        xfield = gamma.field
        zero = xfield.zero()
        one = xfield.one()

        # prepare loop
        table_extension = []
        input_running_evaluation = zero
        input_running_indeterminate = one

        # loop over all rows of table
        for row in input_table.table:
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # match with this:
            # 3. evaluation for input
            # if row[current_instruction] == BaseFieldElement(ord(','), field):
            #    input_evaluation += input_indeterminate * row[memory_value]
            #    input_indeterminate *= gamma
            #new_row += [input_indeterminate]
            #new_row += [input_evaluation]

            new_row += [input_running_indeterminate]
            new_row += [input_running_evaluation]

            input_running_evaluation += input_running_indeterminate * new_row[0]
            input_running_indeterminate *= gamma

            table_extension += [new_row]

        extended_input_table = IOExtension(gamma)
        extended_input_table.table = table_extension

        return extended_input_table

    def transition_constraints(self):
        input_, indeterminate, evaluation, \
            input_next, indeterminate_next, evaluation_next = MPolynomial.variables(6, self.field)
        
        polynomials = []

        polynomials += [indeterminate_next - indeterminate * self.gamma]
        polynomials += [evaluation + indeterminate * input_ - evaluation_next]

        return polynomials
    
    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[1] - one), # indeterminate
                (0, x[2] - zero)] # evaluation