## Welcome to **[This Project ...](https://)**

This notebook is associated with the project submission of MOHAMMED RASHED and GERALD MCCOLLAM and for the class `EN.525.733.8VL.SP22 Deep Learning for Computer Vision`. Our semester project is [Weed Object Detection](https://).

<hr>

<b>DATA: Deep weeds data load (local Machine)</b>

The load_deepweeds_local is used for loading the deepweeds data intoPytorch dataload format into "data/train/[class label]/", "data/val/[class label]/" and ""data/val/[class label]/" directories.


<b>DCGAN</b>

i. DCGAN.ipynb: the DCGAN notebook is used for training the generator/discriminator. The Notebook requires the data file to be in the same directory as the notebook. The images should be stored as "data/train/[class label]/" directory structure. The notebook will also save the generator model in "MOdel" folder. 
Dependency for DCGAN: dcgan_network.py in the same directory.

ii. DCGAN Evaluation.ipynb: this notebook is used for evaluating the DCGAN generator model. The notebook will require a trained generator model file "Model/deepweeds_DCGAN_gan_0.pth" to exisit in the model folder. Please ensure. The DCGAN.ipynb must be run first if the trained model does not exisit. 



