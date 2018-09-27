import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import random
from PIL import Image
# number 1 to 10 data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
 
def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    
    initial = tf.random_uniform(
    shape,
    minval=-0.05,
    maxval=0.05,
    dtype=tf.float32,
)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
 
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=([1, 1, 1, 1]), padding='SAME')
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



 
xs = tf.placeholder(tf.float32, shape=(None,101,101,3),name="xs") 
ys = tf.placeholder(tf.float32, shape=(None,15),name="ys")

keep_prob = tf.placeholder(tf.float32)
 
x_image = tf.reshape(xs, [-1, 101, 101, 3])
 
 
## conv1 layer ##
W_conv1 = weight_variable([4,4, 3,32]) # patch 4x4, in size 3, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 101x101x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 51x51x32

## conv2 layer ##
W_conv2 = weight_variable([4,4, 32, 32]) # patch 4x4, in size 32, out size 32
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 51x51x32
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 26x26x32

## conv3 layer ##
W_conv3 = weight_variable([4,4, 32,32]) # patch 4x4, in size 1, out size 32
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # output size 26x26x32
h_pool3 = max_pool_2x2(h_conv3)                                         # output size 13x13x32


## conv4 layer ##
W_conv4 = weight_variable([4,4, 32, 32]) # patch 4x4, in size 32, out size 32
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv2) + b_conv2) # output size 13x13x32
h_pool4 = max_pool_2x2(h_conv4)                                         # output size 7x7x32

## func1 layer ##
W_fc1 = weight_variable([7*7*32, 200])
b_fc1 = bias_variable([200])
# [n_samples, 7, 7, 32] ->> [n_samples, 7x7x32]
 
h_pool4_flat = tf.reshape(h_pool4, [-1, 7*7*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([200, 15])
b_fc2 = bias_variable([15])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='out')
 

 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))# loss
train_step = tf.train.AdamOptimizer(1*(1e-5)).minimize(cross_entropy)
tf.add_to_collection("train_step", train_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
 

img_test = []
label_test = []
fp = open('little/test.txt', "r")

line = fp.readline()
while line:
    lab = np.zeros(15,dtype = np.float32)
    mid = line.find(".jpg")+4
    lab[int(line[mid:len(line)])] = 1
    img = cv2.imread(line[0:mid])
    img = cv2.resize(img,dsize=(101,101))
    img_test.append(img)
    label_test.append(lab)
    line = fp.readline()
img_test = np.asarray(img_test,dtype=np.float32)
label_test = np.asarray(label_test)


img_list = []
label_list = []
fp = open('little/train.txt', "r")
line = fp.readline()
while line:
    lab = np.zeros(15,dtype = np.float32)
    mid = line.find(".jpg")+4
    lab[int(line[mid:len(line)])] = 1
    img = cv2.imread(line[0:mid])
    img = cv2.resize(img,dsize=(101,101))
    img_list.append(img)
    label_list.append(lab)
    line = fp.readline()
img_list = np.asarray(img_list,dtype=np.float32)
label_list = np.asarray(label_list)


dataset = tf.data.Dataset.from_tensor_slices((img_list,label_list))
dataset = dataset.batch(100)
dataset = dataset.shuffle(buffer_size=1)

iter = dataset.make_initializable_iterator()
el = iter.get_next()

for i in range(1):
    sess.run(iter.initializer)
    for j in range(int(len(label_list)/100)):
        img , label = sess.run(el)
        sess.run(train_step, feed_dict={xs:img, ys:label, keep_prob: 0.8})
    print(compute_accuracy(img_test, label_test))
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, "./model_name")
saver.export_meta_graph("./model_name.meta")
