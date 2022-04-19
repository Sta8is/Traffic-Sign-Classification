# Traffic-Sign-Classification

This repository contains the 3rd and 4th project done during the 7th semester university subject "Computer Vision".
## Traditional ML Algorithms
The 3rd task was to classify part of the [Belgium TS(Traffic Sign) Dataset](https://btsd.ethz.ch/shareddata/). More specifically 34 of total 62 classes used. This dataset is characterised for class inbalance. 
- First of all, SIFT detector and descriptor is used to extract local features. 
- The second step is to build the Bug Of Visual Words model. In order to do that a clustering algorithm is used in order to create a dictionary of global visual features. In this project K-Means algorithm is used.
- Finally the following Classifiers are used: a) Support Vector Machines (One vs all) b)K nearest neighbors

## Deep Learning Techniques
The 4rth task was to create 2 Neural Networks for classification on the same dataset. One neural network will be trained from scratch while the other will be pretrained with a custom classifier(frozen backbone and trainable classifier). In this project my custom CNN achieved 97.44% accuracy. 

Imagedb and imagedb_test are the dataset folders.
For all methods many hyperparameter comparisons were made. In addition different distance metrics were tested(For example in K-NN classifier).
In this project python with OpenCV(3rd task) and Tensorflow-Keras(4th task) are used. Reports are in greek.
