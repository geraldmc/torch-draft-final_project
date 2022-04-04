#hide
from fastbook import *
LABEL_DIRECTORY = "./labels/"
IMG_DIRECTORY = "./images/"

train_dataframe = pd.DataFrame()
val_dataframe = pd.DataFrame()
test_dataframe = pd.DataFrame()

# Prepare training, validation and testing labels for 5 fold
for k in range(2):
    train_label_file = "{}train_subset{}.csv".format(LABEL_DIRECTORY, k)
    val_label_file = "{}val_subset{}.csv".format(LABEL_DIRECTORY, k)
    test_label_file = "{}test_subset{}.csv".format(LABEL_DIRECTORY, k)
    train_dataframe = pd.concat([train_dataframe, 
                                 pd.read_csv(train_label_file)])
    val_dataframe = pd.concat([val_dataframe, 
                               pd.read_csv(val_label_file)])
    test_dataframe = pd.concat([test_dataframe, pd.read_csv(test_label_file)])
    
#name colummns for validation 
train_dataframe["is_valid"]  = False
val_dataframe["is_valid"] = True

###combine train and valid into one dataframe
train_df = pd.concat([train_dataframe,val_dataframe])
train_df.shape

#save the combined test dataframe for future use

# Construct the Datablock

def get_x(r): return IMG_DIRECTORY + r['Filename']
def get_y(r): return r['Label']
dblock = DataBlock(get_x = get_x, get_y = get_y)
dsets = dblock.datasets(train_df)
dsets.train[10]

#check to see if the right image has been entered from the original csv
train_dataframe[train_dataframe.Filename == '20180109-091440-1.jpg']

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y)

dsets = dblock.datasets(train_df)
dsets.train[0]

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms = RandomResizedCrop(128, min_scale=0.35))
dls = dblock.dataloaders(train_df, bs=64)

dls.show_batch(nrows=3, ncols=3)

# Do a training run that will serve as a baseline:

model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)

# Improvement - normalization¶
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms=Resize(128),
                   batch_tfms=[*aug_transforms(size=64, min_scale=0.75),
                               Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(train_df, bs=64)

X = torch.solve(B, A).solution
should be replaced with
X = torch.linalg.solve(A, B) (Triggered internally at  ..\aten\src\ATen\native\BatchLinearAlgebra.cpp:766.)
  ret = func(*args, **kwargs)

model = xresnet50(n_out=dls.c, pretrained=True)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)

