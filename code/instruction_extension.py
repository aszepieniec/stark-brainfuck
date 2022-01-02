from instruction_table import *

class InstructionExtension(InstructionTable):
    def __init__( self, challenges ):
        field = challenges[0].field

        # names for challenges
        challenges = [MPolynomial.constant(c) for c in challenges]
        self.a = challenges[0]
        self.b = challenges[1]
        self.c = challenges[2]
        self.alpha = challenges[6]
        self.eta = challenges[10]

        super(InstructionExtension, self).__init__(field)
        self.width = 2+1+1
    
    @staticmethod
    def extend( instruction_table, challenges):

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
        eta = challenges[10]

        # algebra stuff
        field = instruction_table.field
        xfield = a.field
        one = xfield.one()

        # prepare loop
        table_extension = []
        instruction_permutation_running_product = one
        subset_running_product = one

        # loop over all rows of table
        for i in range(len(instruction_table.table)):
            row = instruction_table.table[i]
            new_row = []

            # first, copy over existing row
            new_row = [xfield.lift(nr) for nr in row]

            # match with this:
            # 1. running product for instruction permutation
            #instruction_permutation_running_product *= alpha - a * row[instruction_pointer] - b * row[current_instruction] - c * row[next_instruction]
            #new_row += [[instruction_permutation_running_product]]

            new_row += [instruction_permutation_running_product]
            current_instruction = instruction_table.table[i][1]
            if i < len(instruction_table.table)-1:
                next_instruction = instruction_table.table[i+1][1]
            else:
                next_instruction = field.zero()
            instruction_permutation_running_product *= alpha - a * \
                new_row[0] - b * xfield.lift(current_instruction) - c * xfield.lift(next_instruction)

            # match with this

            # ifnewaddress = address_next - address
            # ifoldaddress = address_next - address - MPolynomial.constant(self.field.one())

            # polynomials += [ifnewaddress *  ( subset * ( self.eta - self.a * address - self.b * instruction ) - subset_next ) \
            #                 + ifoldaddress * ( subset - subset_next ) ]
            new_row += [subset_running_product]
            if i < len(instruction_table.table) - 1 and instruction_table.table[i+1][0] != instruction_table.table[i][0]:
                subset_running_product *= eta - a * xfield.lift(instruction_table.table[i][0]) - b * xfield.lift(instruction_table.table[i][1])

            table_extension += [new_row]

        extended_instruction_table = InstructionExtension(challenges)
        extended_instruction_table.table = table_extension

        return extended_instruction_table

    def transition_constraints(self):
        address, instruction, permutation, subset, \
             address_next, instruction_next, permutation_next, subset_next = MPolynomial.variables(8, self.field)
        
        polynomials = InstructionExtension.transition_constraints_afo_named_variables(address, instruction, address_next, instruction_next)

        polynomials += [permutation * \
                            ( self.alpha - self.a * address \
                                - self.b * instruction \
                                - self.c * instruction_next ) \
                             - permutation_next]

        ifnewaddress = address_next - address
        ifoldaddress = address_next - address - MPolynomial.constant(self.field.one())

        polynomials += [ifnewaddress * \
                            ( \
                                subset * \
                                ( \
                                    self.eta \
                                    - self.a * address \
                                    - self.b * instruction \
                                ) \
                                - subset_next \
                            ) \
                        + ifoldaddress * \
                            ( \
                                subset - subset_next
                            )]

        return polynomials
    
    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        return [(0, x[0] - zero), # address starts at zero
                (0, x[2] - one)] # running product starts at alpha - a * addr - b * instr - c * instr_next