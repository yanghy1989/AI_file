import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt 
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

tf.reset_default_graph()
## define the train data 
x=tf.placeholder(tf.float32,[None,784]) #input data shape is 28*28
y=tf.placeholder(tf.float32,[None,10])  #the label is 0-9
 # the train parameters
W=tf.Variable(tf.random_normal(([784,10])))
b=tf.Variable(tf.zeros([10]))

#define the output

pred=tf.nn.softmax(tf.matmul(x,W)+b)


saver=tf.train.Saver()
model_path="log/model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,model_path)
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))


    output=tf.argmax(pred,1)
    batch_xs,batch_ys=mnist.train.next_batch(2)
    output_val,predv=sess.run([output,pred],feed_dict={x:batch_xs})
    print(output_val,predv,batch_ys)

    im=batch_xs[1]
    im=im.reshape(-1,28)
    plt.imshow(im)
    plt.show()
