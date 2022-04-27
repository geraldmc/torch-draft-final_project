## Welcome to **[This Project ...](https://)**

This notebook is associated with the project submission of MOHAMMED RASHED and GERALD MCCOLLAM and for the class `EN.525.733.8VL.SP22 Deep Learning for Computer Vision`. Our semester project is [Weed Object Detection](https://).

<hr>

-------------------------------------------------------GAN SECTION-----------------------------------------------------------------------------------------

<b>DATA: Deep weeds data load (local Machine)</b>

i) load_deepweeds_local.ipnb: The load_deepweeds_local is used for loading the deepweeds data intoPytorch dataload format into "data/train/[class label]/", "data/val/[class label]/" and ""data/val/[class label]/" directories.

ii) DataAugmentation.ipynb: this notebook is used for augmenting the training data. Augmented training data is saved in "data/train_dataaug" folder. 

iii) Generate_ACGAN_Images.ipynb: this notebook is used to generate ACGAN images and save in "data/test_generated/[class label]" folder.

<br/>

<b>DCGAN</b>

i. DCGAN.ipynb: the DCGAN notebook is used for training the generator/discriminator. The Notebook requires the data file to be in the same directory as the notebook. The images should be stored as "data/train/[class label]/" directory structure. The notebook will also save the generator model in "Model" folder. 
Dependency for DCGAN: dcgan_network.py in the same directory.

ii. DCGAN Evaluation.ipynb: this notebook is used for evaluating the DCGAN generator model. The notebook will require a trained generator model file "Model/deepweeds_DCGAN_gan_0.pth" to exisit in the model folder. The DCGAN.ipynb must be run first if the trained model does not exisit. 


<br/>

<b>ACGAN</b>

i. AC-GAN.ipynb: the DCGAN notebook is used for training the generator/discriminator. The Notebook requires the data file to be in the same directory as the notebook. The images should be stored as "data/trin/[class label]/" directory structure. Inorder to use train the model on augmented data, first run the DataAugmentation.ipynb. The train data for augmented data is in "data/train_dataaug/[class label]/". The notebook will also save the generator model in "Model" folder. 
Dependency for AC-GAN.ipynb: acgan_network.py in the same directory.

ii. ACGAN Evaluation.ipynb: this notebook is used for evaluating the ACGAN generator model. The notebook will require a trained generator model file ""Model/deepweeds_acgan_gan_v3.pth"" to exisit in the model folder. The AC-GAN.ipynb must be run first if the trained model does not exisit. 
