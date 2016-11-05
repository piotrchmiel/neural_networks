import os
import matplotlib.pyplot as plt
import numpy as np
import re
import urllib.request


class ProcessUJIDataset(object):

    def __init__(self):
        if not ProcessUJIDataset.is_dataset_present():
            ProcessUJIDataset.acquire_dataset()

        self.letters = []

    def is_dataset_present(self):
        pass

    def acquire_dataset(self):
        pass

    def read_available_letters(self):
        pass

    def extract_letters(self):
        pass


class ProcessUJI1(ProcessUJIDataset):
    def __init__(self):
        super.__init__(self)
        self.persons_count = 11
        self.persons = range(1, self.persons_count + 1)
        self.data = []

    def is_dataset_present(self):
        return [os.path.exists(file) for file in \
                ["%s/datasets/UJIpenchars-w%02d" % (os.path.realpath(__file__), i) for i in self.persons]]. \
                   count(False) == 0

    def acquire_dataset(self):
        for i in self.persons:
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version1/UJIpenchars-w%02d" %
                i, os.path.join(os.path.realpath(__file__), "datasets", "UJIpenchars-w%02d" % i))

    def read_available_letters(self):
        letters_regex = re.compile('^.LEXICON')
        letter_regex = re.compile('"([a-zA-Z0-9])"')
        with open(os.path.join(os.path.realpath(__file__), "datasets", "UJIpenchars-w01"), 'r', encoding="utf8") as \
                dataset_file:
            line = dataset_file.readline()
            if not line:
                return
            if letters_regex.search(line):
                self.letters = re.findall(letter_regex, line)
                return

    # source: https://nineties.github.io/prml-seminar/prog/prog21-2.py
    @staticmethod
    def read_traj_data(person, char):
        # 一人あたり2データ
        traj = [[], []]
        f = open(os.path.join(os.path.realpath(__file__), "datasets", "UJIpenchars-w%02d" % person))
        # .SEGMENT CHARACTER ... という行を見つける
        pat = re.compile('.SEGMENT CHARACTER[^?]*\? "%s"' % char)
        cnt = 0
        while True:
            line = f.readline()
            if not line: break
            result = pat.search(line)
            if result:
                f.readline()
                f.readline()
                f.readline()
                while True:
                    line = f.readline().strip()
                    if line == '.PEN_UP':
                        break
                    traj[cnt].append(line.split())
                traj[cnt] = np.array(traj[cnt]).astype(np.float)
                cnt += 1
        f.close()
        return traj

    def extract_letters(self):
        for letter in self.letters:
            for person in self.persons:
                self.data += self.read_traj_data(person, letter)

    # source same as in read_traj_data
    def dump_letters(self):
        plt.figure(figsize=(1, 1), dpi=80, )
        for i in range(22):
            x = self.data[i][:, 0]
            y = self.data[i][:, 1]
            plt.clf()
            plt.axis('off')
            plt.gca().invert_yaxis()
            plt.scatter(x, y, s=0)
            plt.plot(x, y, '-')
            # FIXME
            plt.savefig(os.path.join(os.path.realpath(__file__), "datasets", "UJI1-%s-%d.png" % (letter, person)))


class ProcessUJI2(ProcessUJIDataset):
    def __init__(self):
        super.__init__(self)

    def is_dataset_present(self):
        return os.path.exists(os.path.join(os.path.realpath(__file__), "datasets", "ujipenchars2.txt"))

    def acquire_dataset(self):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version2/ujipenchars2.txt",
            os.path.join(os.path.realpath(__file__), "datasets", "ujipenchars2.txt"))

    def read_available_letters(self):
        letters_regex = re.compile('// ASCII char: ([a-zA-Z0-9])')
        with open(os.path.join(os.path.realpath(__file__), "datasets", "ujipenchars2.txt"), 'r', encoding="utf8") as \
                dataset_file:
            line = dataset_file.readline()
            if not line:
                return
            if letters_regex.search(line):
                letter = letters_regex.findall(line)[0]
                if not letter in self.letters:
                    self.letters.append(letter)


class ProcessDatasets(object):
    def __init__(self):
        self.uji1 = ProcessUJI1()
        self.uji2 = ProcessUJI2()

    def process_uji_1_dataset(self):
        pass

    def process_uji_2_dataset(self):
        pass
