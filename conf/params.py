# import the necessary packages
import torch
import os

# define the root data directory, train and test dataset paths
DATA_PATH = 'data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')

# specify image size and batch size
IMAGE_SIZE = 300
PRED_BATCH_SIZE = 4

# specify threshold confidence value for ssd detections
THRESHOLD = 0.50

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define the other paths 

GD_ZIP_IMG = '/content/gdrive/MyDrive/DL4CV-2022/project-II/data/images.zip'
GD_ZIP_MODEL = '/content/gdrive/MyDrive/DL4CV-2022/project-II/data/models.zip'
IMG_DIRECTORY = '/content/torch-draft-final_project-main/data/images'
IMG_ZIP_FILE = '/content/torch-draft-final_project-main/data/images.zip'
MODEL_ZIP_FILE = '/content/torch-draft-final_project-main/data/models.zip'

OUTPUT_PATH = 'output'
LABEL_PATH = os.path.join(DATA_PATH, 'labels')
MODEL_PATH = os.path.join(DATA_PATH, 'models')
IMAGE_PATH = os.path.join(DATA_PATH, 'images')
IMAGE_ZIP_FILE = os.path.join(IMAGE_PATH, 'images.zip')
MODEL_GD_ID = ''
IMAGE_GD_ID = ''
YOLO_OUTPUT = os.path.join(OUTPUT_PATH, 'yolo_output')