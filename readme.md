# tfcc_image_classify

Load images from disk and train a classifier with TensorFlow_cc.

Up to now, **the fc model is working, but the cnn model is under maintenance.**

## DataSet

Mnist .png images.

- ./data/train: 60000 images.
- ./data/test: 10000 images.

## To run

1. [compilling tensorflow source code into C++ library file](https://github.com/hemajun815/tutorial/blob/master/tensorflow/compilling-tensorflow-source-code-into-C++-library-file.md)
2. `make` or `make run`