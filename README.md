# pix2pix-GANs

We are building pix2pix GANs. For this we would be using PyTorch.
We would be using Satellite-Map Image dataset(http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz).

For more detailed explaination you may use this [Blog](https://medium.com/@Skpd/pix2pix-gan-for-generating-map-given-satellite-images-using-pytorch-6e50c318673a)

### Model Architecture
Since pix2pix is a GAN based architecture, we have one generator which is generating images give some "input", and one discriminator which would discriminate the given image as real or fake. 
pix2pix is best for Image-to-Image translation, where we have one image from one domain and another image of different domain. Our generator will try to come up with  image from second domain given an image from domain one.

Generator architecture is similar to an autoencoder model, where as discriminator's architecture is similar to a binary classifier.  
 
 Generator's architecture is similar to U-Net architecture.
 ![U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png) 
In case of discriminator, it is a patch-wise discriminator. The input given to the discriminator is the concatenation  of Image from domain 1 and generated image of domain 2. 
![Model Architectures](https://drive.google.com/uc?export=view&id=1011ku_4XxVHKDX4X6WNGH3oK6EbtgzJg)
Generator and Discriminator is being done in [Models.py](https://github.com/shashi7679/pix2pix-GANs/blob/master/Models.py)

### Dataset Prepration
Since we are working on Satellite Image to Map generator, the dataset which is available consists of the image both satellite image and respective map image side by side. 
![Sample 1](https://drive.google.com/uc?export=view&id=1MMEnfkzb-b4oE_gwdeccpxzkGQhPfVl1)Each image in the dataset is of shape (1200, 600, 3). So first we need split the image in that format so that, the dataloader gets the that in (satellite_image, map_image) format. We are also doing basic augmentation to the input in order to make it more our generator more robust.
Dataprepration is being done in [dataset.py](https://github.com/shashi7679/pix2pix-GANs/blob/master/dataset.py)

### Hyperparameters
|  Hyperparametrs  | Value  |
|--|--|
|  Learning Rate|2e-4  |
| beta1| 0.5|
|Batch Size | 16|
|Number of workers | 2|
| Image Size| 256|
| L1_Lambda| 100|
| Lambda_GP| 10|
| Epochs| 800|

Configuration of these hyperparameters is being done in [config.py](https://github.com/shashi7679/pix2pix-GANs/blob/main/config.py)

## Training Results
#### _After 1st Epoch_
![Output after Epoch 1](https://drive.google.com/uc?export=view&id=1bQ6xXN8jEWb14BpKjWBrnWInbyrJLxPi)_Satellite Image(left), Map(middle), Generated Map(right)_

#### _After 100 Epochs_
![Output after Epoch 100](https://drive.google.com/uc?export=view&id=1QNGRz16127euLiQ_BxcA_F9xPLu2mQZY)_Satellite Image(left), Map(middle), Generated Map(right)_

#### _After 400 Epochs_
![Output after Epoch 400](https://drive.google.com/uc?export=view&id=1bddfataLUOTYyH7E1iq6B-XY255lnQ_O)_Satellite Image(left), Map(middle), Generated Map(right)_

#### _After 800 Epochs_
![Output after Epoch 800](https://drive.google.com/uc?export=view&id=16GbzMbWrOFfP3kO4h9wPvBnpHxaPlNsU)_Satellite Image(left), Map(middle), Generated Map(right)_
### Generator Loss Vs. Discriminator Loss
![Generator Loss Vs. Discriminator Loss](https://drive.google.com/uc?export=view&id=19FnwwJN1gtnCK2Y6XlhAH_YGvUB7rmJB)
## Training
```sh
bash download.sh
git clone git@github.com:shashi7679/pix2pix-GANs.git
cd pix2pix-GANs
```
Run train.ipynb on Jupyter Notebook
- For training, set LOAD_MODEL as False and SAVE_MODEL as True in [config.py](https://github.com/shashi7679/pix2pix-GANs/blob/master/config.py)
- For Validation/ Using the saved model, set LOAD_MODEL as True in [config.py](https://github.com/shashi7679/pix2pix-GANs/blob/master/config.py). 
- To download the pretrained models of validation [Click Here](https://drive.google.com/drive/folders/1jgqB6zVJ3iSXyQ8JdrikTerk74DkTPSY?usp=sharing)


## References
- https://arxiv.org/abs/1611.07004
