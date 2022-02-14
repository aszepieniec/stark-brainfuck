class LabeledList:
    def __init__(self):
        self.objects = []

    def add(self, raw_list, common_label):
        for i in range(len(raw_list)):
            self.append(raw_list[i], common_label + ":" + str(i))

    def bare(self):
        bare_list = [o[0] for o in self.objects]
        return bare_list

    def append(self, object, label):
        self.objects += [(object, label + " / " + str(len(self.objects)))]

    def concatenate(self, other):
        self.objects += other.objects

    def __len__(self):
        return len(self.objects)

    def get(self, index):
        return self.objects[index][0]

    def label(self, index):
        return self.objects[index][1]
