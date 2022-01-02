from io_table import *

class IOExtension(IOTable):
    def __init__(self, gamma):
        field = gamma.field

        # names for challenges
        self.gamma = MPolynomial.constant(gamma)

        super(IOExtension, self).__init__(field)
        self.width = 1+1

    @staticmethod
    def extend(input_table, gamma):

        # algebra stuff
        xfield = gamma.field
        zero = xfield.zero()
        one = xfield.one()

        # prepare loop
        table_extension = []
        input_running_evaluation = zero

        # loop over all rows of table
        for row in input_table.table:
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # match with this:
            # 3. evaluation for input
            # if row[current_instruction] == BaseFieldElement(ord(','), field):
            #    input_evaluation += input_evaluation * gamma + row[io_value]
            #new_row += [input_evaluation]

            new_row += [input_running_evaluation]

            input_running_evaluation = input_running_evaluation * gamma + new_row[0]

            table_extension += [new_row]

        extended_input_table = IOExtension(gamma)
        extended_input_table.table = table_extension

        return extended_input_table

    def transition_constraints(self):
        input_, evaluation, \
            input_next, evaluation_next = MPolynomial.variables(6, self.field)
        
        polynomials = []

        polynomials += [evaluation * self.gamma + input_ - evaluation_next]

        return polynomials
    
    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        zero = MPolynomial.zero()
        return [(0, x[1] - zero)] # evaluation