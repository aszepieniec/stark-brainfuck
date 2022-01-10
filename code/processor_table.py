from aet import *


class ProcessorTable(Table):
    # column (=register) names
    cycle = 0
    instruction_pointer = 1
    current_instruction = 2
    next_instruction = 3
    memory_pointer = 4
    memory_value = 5
    is_zero = 6

    def __init__(self, field):
        super(ProcessorTable, self).__init__(field, 7)

    def pad( self ):
        while len(self.table) & (len(self.table)-1) != 0:
            new_row = [self.field.zero()] * 7
            new_row[ProcessorTable.cycle] = self.table[-1][ProcessorTable.cycle] + self.field.one()
            new_row[ProcessorTable.instruction_pointer] = self.table[-1][ProcessorTable.instruction_pointer]
            new_row[ProcessorTable.current_instruction] = self.field.zero()
            new_row[ProcessorTable.next_instruction] = self.field.zero()
            new_row[ProcessorTable.memory_pointer] = self.table[-1][ProcessorTable.memory_pointer]
            new_row[ProcessorTable.memory_value] = self.table[-1][ProcessorTable.memory_value]
            new_row[ProcessorTable.is_zero] = self.table[-1][ProcessorTable.is_zero]
            self.table += [new_row]

    @staticmethod
    def if_instruction(instruction, indeterminate: MPolynomial):
        '''if_instruction(instr, X)
        returns a polynomial in X that evaluates to 0 in X=FieldElement(instr)'''
        field = list(indeterminate.dictionary.values())[0].field
        return MPolynomial.constant(field(ord(instruction))) - indeterminate

    @staticmethod
    def ifnot_instruction(instruction, indeterminate: MPolynomial):
        '''ifnot_instruction(instr, X)
        returns a polynomial in X that evaluates to 0 in all instructions except for X=FieldElement(instr)'''
        field = list(indeterminate.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())
        acc = one
        for c in "[]<>,.+-":
            if c != instruction:
                acc *= indeterminate - \
                    MPolynomial.constant(field(ord(c)))
        return acc

    @staticmethod
    def instruction_polynomials(instr, cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, is_zero, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, is_zero_next):
        zero = MPolynomial.zero()
        field = list(cycle.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())
        two = MPolynomial.constant(field.one()+field.one())
        polynomials = [zero] * 3

        if instr == '[':
            polynomials[ProcessorTable.cycle] = memory_value * (instruction_pointer_next - instruction_pointer - two) + \
                is_zero * (instruction_pointer_next - next_instruction)
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
            polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value

        elif instr == ']':
            polynomials[ProcessorTable.cycle] = is_zero * (instruction_pointer_next - instruction_pointer - two) + \
                memory_value * (instruction_pointer_next - next_instruction)
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
            polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value

        elif instr == '<':
            polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer + one
            # memory value, satisfied by permutation argument
            polynomials[ProcessorTable.current_instruction] = zero

        elif instr == '>':
            polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer - one
            # memory value, satisfied by permutation argument
            polynomials[ProcessorTable.current_instruction] = zero

        elif instr == '+':
            polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
            polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value - one

        elif instr == '-':
            polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
            polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value + one

        elif instr == ',':
            polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
            # memory value, set by evaluation argument
            polynomials[ProcessorTable.current_instruction] = zero

        elif instr == '.':
            polynomials[ProcessorTable.cycle] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[ProcessorTable.instruction_pointer] = memory_pointer_next - memory_pointer
            polynomials[ProcessorTable.current_instruction] = memory_value_next - memory_value

        # account for padding:
        # deactivate all polynomials if current instruction is zero
        for i in range(len(polynomials)):
            polynomials[i] *= current_instruction

        return polynomials

    def transition_constraints_afo_named_variables(self, cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, is_zero, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, is_zero_next):
        one = MPolynomial.constant(self.field.one())

        polynomials = [MPolynomial.zero()] * 3

        # instruction-specific polynomials
        for c in "[]<>+-,.":
            instr = ProcessorTable.instruction_polynomials(c,
                                                      cycle,
                                                      instruction_pointer,
                                                      current_instruction,
                                                      next_instruction,
                                                      memory_pointer,
                                                      memory_value,
                                                      is_zero,
                                                      cycle_next,
                                                      instruction_pointer_next,
                                                      current_instruction_next,
                                                      next_instruction_next,
                                                      memory_pointer_next,
                                                      memory_value_next,
                                                      is_zero_next)
            deselector = self.ifnot_instruction(c, current_instruction)

            for i in range(len(instr)):
                polynomials[i] += deselector * instr[i]

        # instruction-independent polynomials
        polynomials += [cycle_next - cycle - one]  # cycle increases by one
        polynomials += [is_zero * memory_value]  # at least one is zero
        polynomials += [is_zero * (one - is_zero)]  # 0 or 1

        return polynomials

    def transition_constraints(self):
        cycle, \
            instruction_pointer, \
            current_instruction, \
            next_instruction, \
            memory_pointer, \
            memory_value, \
            is_zero, \
            cycle_next, \
            instruction_pointer_next, \
            current_instruction_next, \
            next_instruction_next, \
            memory_pointer_next, \
            memory_value_next, \
            is_zero_next = MPolynomial.variables(14, self.field)

        return self.transition_constraints_afo_named_variables(cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, is_zero, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, is_zero_next)

    def boundary_constraints(self):
        # format: (cycle, polynomial)
        x = MPolynomial.variables(self.width, self.field)
        one = MPolynomial.constant(self.field.one())
        zero = MPolynomial.zero()
        constraints = [(0, x[ProcessorTable.cycle] - zero),
                       (0, x[ProcessorTable.instruction_pointer] - zero),
                       # (0, ???), # current instruction
                       # (0, ???), # next instruction
                       (0, x[ProcessorTable.memory_pointer] - zero),
                       (0, x[ProcessorTable.memory_value] - zero),
                       (0, x[ProcessorTable.is_zero] - one)] 

        return constraints