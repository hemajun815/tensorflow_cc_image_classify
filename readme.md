# tfcc_image_classify

Load images from disk and train a classifier with TensorFlow_cc.

Up to now, I applied the gradients by using 'AddSymbolicGradients' and 'ApplyGradientDescent' in the fully connection layer, but I have not found the right way to apply the gradients in the convolution layer. So **the fc model is working, but the cnn model is under maintenance.** I hope that who can give me some advice about how to apply the gradients in the convolution layer.

## DataSet

Mnist .png images.

- ./data/train: 60000 images.
- ./data/test: 10000 images.

## To run

1. [compilling tensorflow source code into C++ library file](https://github.com/hemajun815/tutorial/blob/master/tensorflow/compilling-tensorflow-source-code-into-C++-library-file.md)
2. `make` or `make run`