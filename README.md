# Learning Based Human-Machine Interaction Coursework

This repository contains the coursework for the "Learning Based Human-Machine Interaction" class. It covers three main areas: Principal Component Analysis (PCA), Neural Networks (NNs), and Convolutional Neural Networks (CNNs).

## Homework 1 - PCA

The PCA homework involves using a training set of images, with each image representing a digit from 0-9. 16 images are selected for each digit to be used as training data for PCA. Similarly, 10 images per digit are selected from the test set images as testing data for PCA. An open-source PCA tool can be used for this exercise.

The tasks involve:

1. Projecting these 100 images into PCA space to obtain PCA projection weights, recording these weights in a binary file, and calculating the storage space needed for these weights and the PCA space. We also need to calculate the compression ratio against the original images.
2. Reconstructing the images from the PCA weight files and calculating the average Peak Signal-to-Noise Ratio (PSNR) for the 100 reconstructed images against the original ones.
3. Varying the number of eigenvectors in PCA space and repeating steps 1 and 2. The results are to be plotted as a line graph.

## Homework 2 - Neural Networks

This homework requires you to download the MNIST dataset and select N images for each digit from the training set images as training data for the NN, and M images for each digit from the test set images as testing data. An open-source NN tool is not allowed.

The tasks involve:

1. Training the network using backpropagation and plotting the error rate curve.
2. Running the testing data through the trained NN and plotting the error rate curve.

## Homework 3 - Convolutional Neural Networks

This task involves designing a CNN architecture similar to the one used for the NN in Homework 2. The hidden layer in the NN is replaced with a convolutional layer and a pooling layer, and the final classification is done using a fully connected layer.

The tasks involve:

1. Training the network using backpropagation and plotting the error rate curve.
2. Running the testing data through the trained CNN and plotting the error rate curve.

## Final Project - Fish Detection

For the final project, we are to use the training data from the `TrainingData` folder to design a deep learning network for fish detection. The network can be a modification of existing architectures.

The tasks involve:

1. Plotting the loss curve for both the training set and validation set during the training process.
2. Using the trained model to mark the location of fish in the TrainingData with a red box and comparing it with the ground truth marked with a blue box.
3. Using the trained model to mark the location of fish in the TestData with a red box and comparing it with the ground truth marked with a blue box.
4. Using the trained model to calculate the mean Average Precision (mAP) at different Intersection over Union (IoU) thresholds on the TestData and plotting the results.

