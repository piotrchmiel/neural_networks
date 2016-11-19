from os import path, makedirs


BASE_DIR = path.dirname(path.dirname(__file__))

LOG_DIR = path.join(BASE_DIR, 'tensorflowlogs')

if not path.exists(LOG_DIR):
    makedirs(LOG_DIR)

OBJECT_DIR = path.join(BASE_DIR, 'tensorflowobjects')

if not path.exists(OBJECT_DIR):
    makedirs(OBJECT_DIR)