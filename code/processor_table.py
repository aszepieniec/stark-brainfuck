from aet import *


class ProcessorTable(Table):
    def __init__(self, field):
        super(ProcessorTable, self).__init__(field, 7)

    @staticmethod
    def if_instruction(instruction, indeterminate: MPolynomial):
        '''if_instruction(instr, X)
        returns a polynomial in X that evaluates to 0 in X=FieldElement(instr)'''
        field = list(indeterminate.dictionary.values())[0].field
        return indeterminate - MPolynomial.constant(BaseFieldElement(ord(instruction), field))

    @staticmethod
    def ifnot_instruction(instruction, indeterminate: MPolynomial):
        '''ifnot_instruction(instr, X)
        returns a polynomial in X that evaluates to 0 in all instructions except for X=FieldElement(instr)'''
        field = list(indeterminate.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())
        acc = one
        for c in "[]<>,.+-":
            if c == instruction:
                pass
            acc *= indeterminate - \
                MPolynomial.constant(BaseFieldElement(ord(instruction), field))
        return acc

    @staticmethod
    def instruction_polynomials(instruction, cycle, instruction_pointer, next_instruction, memory_pointer, memory_value, is_zero, cycle_next, instruction_pointer_next, memory_pointer_next, memory_value_next):
        zero = MPolynomial.zero()
        field = list(cycle.dictionary.values())[0].field
        one = MPolynomial.constant(field.one())
        two = MPolynomial.constant(field.one()+field.one())
        polynomials = [zero] * 3

        if instruction == '[':
            polynomials[0] = memory_value * (instruction_pointer_next - instruction_pointer - two) + \
                is_zero * (instruction_pointer_next - next_instruction)
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value

        elif instruction == ']':
            polynomials[0] = is_zero * (instruction_pointer_next - instruction_pointer - two) + \
                memory_value * (instruction_pointer_next - next_instruction)
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value

        elif instruction == '<':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer + one
            # memory value, satisfied by permutation argument
            polynomials[2] = zero

        elif instruction == '>':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer - one
            # memory value, satisfied by permutation argument
            polynomials[2] = zero

        elif instruction == '+':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value - one

        elif instruction == '-':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value + one

        elif instruction == ',':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            # memory value, set by evaluation argument
            polynomials[2] = zero

        elif instruction == '.':
            polynomials[0] = instruction_pointer_next - \
                instruction_pointer - one
            polynomials[1] = memory_pointer_next - memory_pointer
            polynomials[2] = memory_value_next - memory_value

        return polynomials

    def transition_constraints_afo_named_variables(self, cycle, instruction_pointer, current_instruction, next_instruction, memory_pointer, memory_value, is_zero, cycle_next, instruction_pointer_next, current_instruction_next, next_instruction_next, memory_pointer_next, memory_value_next, is_zero_next):
        one = MPolynomial.constant(self.field.one())

        polynomials = [MPolynomial.zero()] * 4

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
                                                      memory_value_next)
            deselector = self.if_instruction(c, current_instruction)
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
        # format: (register, cycle, value)
        constraints = [(0, 0, self.field.zero()),  # cycle
                       # instruction pointer
                       (1, 0, self.field.zero()),
                       # (2, 0, ???), # current instruction
                       # (3, 0, ???), # next instruction
                       (4, 0, self.field.zero()),  # memory pointer
                       (5, 0, self.field.zero()),  # memory value
                       (6, 0, self.field.one())]  # memval==0

        return constraints