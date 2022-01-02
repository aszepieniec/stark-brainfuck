from output_table import *

class OutputExtension(OutputTable):
    def __init__(self, delta):
        field = delta.field

        # names for challenges
        self.delta = MPolynomial.constant(delta)

        super(OutputExtension, self).__init__(field)
        self.width = 1+2

    @staticmethod
    def extend(output_table, delta):

        # algebra stuff
        xfield = delta.field
        zero = xfield.zero()
        one = xfield.one()

        # prepare loop
        table_extension = []
        output_running_evaluation = zero
        output_running_indeterminate = one

        # loop over all rows of table
        for row in output_table.table:
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # match with this:
            # 4. evaluation for output
            # if row[current_instruction] == BaseFieldElement(ord('.'), field):
            #    output_evaluation += output_indeterminate * row[memory_value]
            #    output_indeterminate *= delta
            #new_row += [output_indeterminate]
            #new_row += [output_evaluation]

            new_row += [output_running_indeterminate]
            new_row += [output_running_evaluation]

            output_running_evaluation += output_running_indeterminate * new_row[0]
            output_running_indeterminate *= delta

            table_extension += [new_row]

        extended_output_table = OutputExtension(delta)
        extended_output_table.table = table_extension

        return extended_output_table

    def transition_constraints(self):
        output, indeterminate, evaluation, \
            output_next, indeterminate_next, evaluation_next = MPolynomial.variables(6, self.field)
        
        polynomials = []

        polynomials += [indeterminate_next - indeterminate * self.delta]
        polynomials += [evaluation + indeterminate * output - evaluation_next]
        
        return polynomials
    
    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[1] - one), # indeterminate
                (0, x[2] - zero)] # evaluation