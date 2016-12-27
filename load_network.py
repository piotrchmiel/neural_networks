import os
import joblib
import numpy as np
from src.utils import load_neural_network
from src.settings import OBJECT_DIR

dataset = joblib.load(os.path.join(OBJECT_DIR, "Mnist"))

with load_neural_network() as nn:
    dataset.show_image(dataset.test[98])
    print("Sample result {}, Should be {}".format(nn.predict(dataset.test[98], dataset.map_result),
                                                  dataset.map_result(np.argmax(dataset.test_labels[98]))))
    print("Accuracy: {}".format(nn.accuracy(dataset)))
