# import the necessary packages
import torch
import os

# define the root data directory, train and test dataset paths
DATA_PATH = 'data'
TEST_PATH = os.path.join(BASE_PATH, 'train')
TEST_PATH = os.path.join(BASE_PATH, 'test')

# specify image size and batch size
IMAGE_SIZE = 300
PRED_BATCH_SIZE = 4

# specify threshold confidence value for ssd detections
THRESHOLD = 0.50

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define paths to save output 
OUTPUT_PATH = 'output'
YOLO_OUTPUT = os.path.join(OUTPUT_PATH, 'yolo_output')