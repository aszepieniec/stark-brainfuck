class EvaluationArgument:
    def __init__(self, challenge_index, terminal_index, symbols):
        self.challenge_index = challenge_index
        self.terminal_index = terminal_index
        self.symbols = symbols

    def compute_terminal(self, challenges):
        iota = challenges[self.challenge_index]
        xfield = iota.field
        acc = xfield.zero()
        for s in self.symbols:
            acc = iota * acc + xfield.lift(s)
        return acc

    def select_terminal(self, terminals):
        return terminals[self.terminal_index]


class ProgramEvaluationArgument:
    def __init__(self, challenge_indices, terminal_index, program):
        self.challenge_indices = challenge_indices
        self.terminal_index = terminal_index
        self.program = program

    def compute_terminal(self, challenges):
        trimmed_challenges = [challenges[i] for i in range(
            len(challenges)) if i in self.challenge_indices]
        a, b, c, eta = trimmed_challenges
        xfield = trimmed_challenges[0].field
        running_sum = xfield.zero()
        previous_address = -xfield.one()
        padded_program = [xfield.lift(p)
                          for p in self.program] + [xfield.zero()]
        for i in range(len(padded_program)-1):
            address = xfield(i)
            current_instruction = padded_program[i]
            next_instruction = padded_program[i+1]
            if previous_address != address:
                running_sum = running_sum * eta + a * address + \
                    b * current_instruction + c * next_instruction
            previous_address = address

        index = len(padded_program)-1
        address = xfield(index)
        current_instruction = padded_program[index]
        next_instruction = xfield.zero()
        running_sum = running_sum * eta + a * address + \
            b * current_instruction + c * next_instruction

        return running_sum

    def select_terminal(self, terminals):
        return terminals[self.terminal_index]
