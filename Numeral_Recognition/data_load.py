import tensorflow as tf 
from matplotlib import pyplot as plt 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

print("input data",mnist.train.images)
print("data shape",mnist.train.images.shape)

print("test images shape",mnist.test.images.shape)
print("validation shape",mnist.validation.images.shape)
im=mnist.train.images[1]
im=im.reshape(-1,28)
plt.imshow(im)
plt.show()
