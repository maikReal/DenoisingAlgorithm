# DenoisingAlgorithm
Current repository represents denoising algorithm.

## Train 

During the training, algorithm use on input noisy sound and only noise. Algorithm try to find in certain noisy sound just noise and then clean the noisy one.

If you want to use your own data, change path to train folder on your own path (BUT KEEP THE STRUCTURE). Remember, you should have train and validation folders for running the training!

After training you will see the MSE metric on validation dataset.

# How To Run?

Main files: <br>
1. main.py - with generator of data and script for starting the training
2. model.py - the UNET model for training
3. config.py - file withh all necessary settings for the code

## Locally

1. Setup all necessary parameteres in `config.py` file
2. Run `main.py` script

After finishing of the model training, you will see the <b>MSE metric</b> for validation dataset

## Docker 

1. docker build --tag denoising_algorithm . ==> for building the image
2. docker run --tag denoising_algorithm ==> for running the image as a container

It will take some time, because I COPY all project to container (`train` folder size is about 2G)

# TODO

1. Rewrite Dockerfile
