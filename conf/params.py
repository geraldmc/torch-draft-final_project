# import the necessary packages
import torch
import os


# specify image size and batch size
IMAGE_SIZE = 300
PRED_BATCH_SIZE = 4

# specify threshold confidence value for ssd detections
THRESHOLD = 0.50

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define the root data directory, train and test dataset paths
DATA_PATH = 'data'

GD_ZIP_IMG = '/content/gdrive/MyDrive/DL4CV-2022/project-II/data/images.zip'
GD_ZIP_MODEL = '/content/gdrive/MyDrive/DL4CV-2022/project-II/data/models.zip'
IMG_DIRECTORY = '/content/torch-draft-final_project-main/data/images'

OUTPUT_PATH = 'output'
LABEL_PATH = os.path.join(DATA_PATH, 'labels')
MODEL_PATH = os.path.join(DATA_PATH, 'models')
IMAGE_PATH = os.path.join(DATA_PATH, 'images')
IMG_TRAIN_PATH = os.path.join(DATA_PATH, 'train')
IMG_VAL_PATH = os.path.join(DATA_PATH, 'val')
IMG_TEST_PATH = os.path.join(DATA_PATH, 'test')
IMG_CLASSES_PATH = os.path.join(DATA_PATH, 'image_classes')
IMG_ZIP_FILE = os.path.join(DATA_PATH, 'images.zip')

# Global variables
RAW_IMG_SIZE = (256, 256)
IMG_SIZE = (224, 224)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
MAX_EPOCH = 200
NUM_EPOCHS = 5
BATCH_SIZE = 32
FOLDS = 5
STOPPING_PATIENCE = 32
LR_PATIENCE = 16
INITIAL_LR = 0.0001
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]

CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']