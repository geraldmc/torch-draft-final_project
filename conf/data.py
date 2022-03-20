# import the necessary packages
from torch.utils.data import DataLoader

def get_dataloader(dataset, batchSize, shuffle=True):
	# create a dataloader and return it
	dataLoader= DataLoader(dataset, batch_size=batchSize,
		shuffle=shuffle)
	return dataLoader

def normalize(image, mean=128, std=128):
    # normalize the SSD input and return it 
    image = (image * 256 - mean) / std
    return image