import abc
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import urllib.request
import zipfile
from scipy.misc import imread
from src.settings import IMAGE_SIDE_PIXELS, DATASETS_DIR
from src.image_utils import open_image, trim_image, resize_image, normalize_image


class ProcessUJIDataset(metaclass=abc.ABCMeta):

    def __init__(self, dpi=IMAGE_SIDE_PIXELS):
        self.data = {}
        self.images_prefix = ''
        self.dpi = dpi

    @abc.abstractmethod
    def is_dataset_present(self):
        raise NotImplementedError

    @abc.abstractmethod
    def acquire_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def read_available_letters(self):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_letters(self):
        raise NotImplementedError

    def dump_letters(self):
        for letter, letters_data in self.data.items():
            i = 0
            for single_letter in letters_data:
                plt.clf()
                plt.axis('off')
                plt.gca().invert_yaxis()
                plt.gcf().set_size_inches(1.0, 1.0)
                for line in single_letter:
                    x = line[:, 0]
                    y = line[:, 1]
                    plt.scatter(x, y, s=0)
                    plt.plot(x, y, '-', c='b')
                if letter.isupper() or letter.isdigit():
                    image_filename = os.path.join(DATASETS_DIR, "%s-%s-%d.png" % (self.images_prefix, letter, i))
                    plt.savefig(os.path.join(image_filename), dpi=self.dpi)
                    image = open_image(image_filename)
                    image = trim_image(image)
                    image = resize_image(image)
                    image.save(image_filename)
                i += 1

    def create_csv(self, out_filename=None):
        if out_filename is None:
            out_filename = "%s.csv" % self.images_prefix
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "datasets", out_filename), 'w') as \
                out:
            out.write("%s;%s\n" % ("symbol", ';'.join(["data%d" % i for i in range(self.dpi ** 2)])))
            for letter_file in glob.glob(os.path.join(DATASETS_DIR, "%s-*.png" % self.images_prefix)):
                letter = os.path.basename(letter_file).split('-')[1]
                image = imread(letter_file, flatten=True)
                image = normalize_image(image)
                out.write("%s;%s\n" % (letter, ';'.join(['%.2f' % num for num in image])))


class ProcessUJI1(ProcessUJIDataset):
    def __init__(self):
        super(ProcessUJI1, self).__init__()
        self.persons_count = 11
        self.persons = range(1, self.persons_count + 1)
        self.images_prefix = "UJI1"

    def is_dataset_present(self):
        return [os.path.exists(file) for file in [os.path.join(DATASETS_DIR, "UJIpenchars-w%02d" % i)
                 for i in self.persons]].count(False) == 0

    def acquire_dataset(self):
        for i in self.persons:
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version1/UJIpenchars-w%02d" % i,
                os.path.join(DATASETS_DIR, "UJIpenchars-w%02d" % i))

    def read_available_letters(self):
        letters_regex = re.compile(r'^.LEXICON')
        letter_regex = re.compile(r'"([a-zA-Z0-9])"')
        with open(os.path.join(DATASETS_DIR, "UJIpenchars-w01"), 'r',
                  encoding="utf8") as dataset_file:
            line = dataset_file.readline()
            if not line:
                return
            if letters_regex.search(line):
                self.data = dict((letter, []) for letter in re.findall(letter_regex, line))
                return

    def extract_letters(self):
        letter_regex = re.compile(r'.SEGMENT CHARACTER[^?]*\? "([a-zA-Z0-9])"')
        for person in self.persons:
            with open(os.path.join(DATASETS_DIR, "UJIpenchars-w%02d" %
                      person)) as person_letters:
                line = person_letters.readline()
                while True:
                    need_reading = True
                    if not line:
                        break
                    if letter_regex.search(line):
                        letter = letter_regex.findall(line)[0]
                        for _ in range(3):
                            line = person_letters.readline()
                        letter_curves = []
                        points = []
                        while True:
                            line = person_letters.readline().strip()
                            if line == '.PEN_UP':
                                x = np.array(points[::2]).astype(np.float)
                                y = np.array(points[1::2]).astype(np.float)
                                points = np.column_stack((x, y))
                                letter_curves.append(points)
                                points = []
                                for _ in range(3):
                                    line = person_letters.readline()
                                if not line or '.SEGMENT' in line:
                                    self.data[letter] += [letter_curves]
                                    need_reading = False
                                    break
                            if len(line.split()) > 1:
                                points += line.split()
                    if need_reading:
                        line = person_letters.readline()


class ProcessUJI2(ProcessUJIDataset):
    def __init__(self):
        super(ProcessUJI2, self).__init__()
        self.images_prefix = "UJI2"

    def is_dataset_present(self):
        return os.path.exists(os.path.join(DATASETS_DIR, "ujipenchars2.txt"))

    def acquire_dataset(self):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version2/ujipenchars2.txt",
            os.path.join(DATASETS_DIR, "ujipenchars2.txt"))

    def read_available_letters(self):
        letters_regex = re.compile(r'// ASCII char: ([a-zA-Z0-9])')
        with open(os.path.join(DATASETS_DIR, "ujipenchars2.txt"), 'r', encoding="utf8") as dataset_file:
            for line in dataset_file:
                if letters_regex.search(line):
                    letter = letters_regex.findall(line)[0]
                    if letter not in self.data.keys():
                        self.data[letter] = []

    def extract_letters(self):
        word_regex = re.compile(r'^WORD ([a-zA-Z0-9])')
        curves_regex = re.compile(r'^\s+NUMSTROKES ([0-9]+)')
        points_regex = re.compile(r'^\s+POINTS ([0-9]+) # (.*)')
        with open(os.path.join(DATASETS_DIR, "ujipenchars2.txt"), 'r', encoding="utf8") as dataset_file:
            for line in dataset_file:
                if word_regex.search(line):
                    if word_regex.search(line):
                        letter = word_regex.findall(line)[0]
                        line = dataset_file.readline()
                        if curves_regex.search(line):
                            curves_count = int(curves_regex.findall(line)[0])
                            letter_curves = []
                            for curve in range(curves_count):
                                line = dataset_file.readline()
                                if points_regex.search(line):
                                    points_count, points = points_regex.findall(line)[0]
                                    points = points.split(' ')
                                    x = np.array(points[::2]).astype(np.float)
                                    y = np.array(points[1::2]).astype(np.float)
                                    points = np.column_stack((x, y))
                                    letter_curves.append(points)
                            self.data[letter] += [letter_curves]


class ProcessHWCR(ProcessUJIDataset):
    # source: https://github.com/sachinkariyattin/HWCR/tree/master/training_set/training_set
    def __init__(self):
        super(ProcessHWCR, self).__init__()
        self.persons_count = 11
        self.persons = range(1, self.persons_count + 1)
        self.images_prefix = "HWCR"

    def is_dataset_present(self):
        return os.path.exists(os.path.join(DATASETS_DIR, "HWCR"))

    def acquire_dataset(self):
        with zipfile.ZipFile(os.path.join(DATASETS_DIR, "HWCR-letters.zip"), "r") as zip_ref:
            zip_ref.extractall(DATASETS_DIR)

    def read_available_letters(self):
        pass

    def extract_letters(self):
        pass

    def dump_letters(self):
        i = 0
        previous_letter = ''
        for letter_file in os.listdir(os.path.join(DATASETS_DIR, "HWCR")):
            filename = os.path.join(DATASETS_DIR, "HWCR", letter_file)
            image = open_image(filename)
            image = resize_image(image)
            letter = letter_file[0].upper()
            if previous_letter != letter:
                previous_letter = letter
                i = 0
            image.save(os.path.join(DATASETS_DIR, "%s-%s-%d.png" % (self.images_prefix, letter, i)))
            i += 1


class ProcessDatasets(object):
    def __init__(self):
        self.classes_to_process = [ProcessUJI1(), ProcessUJI2(), ProcessHWCR()]
        for class_ in self.classes_to_process:
            self.process_uji_dataset(class_)

    @staticmethod
    def process_uji_dataset(uji):
        if not uji.is_dataset_present():
            uji.acquire_dataset()
        uji.read_available_letters()
        uji.extract_letters()
        uji.dump_letters()
        uji.create_csv()
