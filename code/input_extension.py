from input_table import *

class InputExtension(InputTable):
    def __init__(self, challenges):
        field = challenges[0].field

        # names for challenges
        challenges = [MPolynomial.constant(c) for c in challenges]
        self.gamma = challenges[8]

        super(InputExtension, self).__init__(field)
        self.width = 1+2

    @staticmethod
    def extend(input_table, challenges):
        # names for challenges
        a = challenges[0]
        b = challenges[1]
        c = challenges[2]
        d = challenges[3]
        e = challenges[4]
        f = challenges[5]
        alpha = challenges[6]
        beta = challenges[7]
        gamma = challenges[8]
        delta = challenges[9]

        # algebra stuff
        field = input_table.field
        xfield = a.field
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

        extended_input_table = InputExtension(challenges)
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