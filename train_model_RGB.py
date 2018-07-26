import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import random
from PIL import Image
# number 1 to 10 data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
#定義準確度函數 如果不懂這部份代碼看完前面MNIST教學 應該就懂了
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
 
#定義初始化變數 採用normal distribution , 標準差為0.1
def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    
    initial = tf.random_uniform(
    shape,
    minval=-0.05,
    maxval=0.05,
    dtype=tf.float32,
)
    return tf.Variable(initial)
 
#定義初始化變數 採用常數 , 皆為為0.0
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
 
#定義conv 層 layer padding 方法採用"一樣"
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=([1, 1, 1, 1]), padding='SAME')
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



 
# 定義placeholder
xs = tf.placeholder(tf.float32, shape=(None,101,101,3)) # 28x28
ys = tf.placeholder(tf.float32, shape=(None,15))
keep_prob = tf.placeholder(tf.float32)
 
#將圖片reshape， -1表示會自動算幾組
#28,28,1 分別代表寬 高 channel數(像RGB的話這裡就要改3)
x_image = tf.reshape(xs, [-1, 101, 101, 3])
 
 
#開始組裝神經網路
## conv1 layer ##
#1:表示 input_size  32:表示output_size 所以這裡表示一張圖總共訓練出32個filter
W_conv1 = weight_variable([4,4, 3,32]) # patch 4x4, in size 3, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 101x101x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 51x51x32

## conv2 layer ##
#這裡表示 一張圖訓練出2個filter
W_conv2 = weight_variable([4,4, 32, 32]) # patch 4x4, in size 32, out size 32
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 51x51x32
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 26x26x32

## conv3 layer ##
#1:表示 input_size  32:表示output_size 所以這裡表示一張圖總共訓練出32個filter
W_conv3 = weight_variable([4,4, 32,32]) # patch 4x4, in size 1, out size 32
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # output size 26x26x32
h_pool3 = max_pool_2x2(h_conv3)                                         # output size 13x13x32


## conv4 layer ##
#這裡表示 一張圖訓練出2個filter
W_conv4 = weight_variable([4,4, 32, 32]) # patch 4x4, in size 32, out size 32
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv2) + b_conv2) # output size 13x13x32
h_pool4 = max_pool_2x2(h_conv4)                                         # output size 7x7x32

## func1 layer ##
W_fc1 = weight_variable([7*7*32, 200])
b_fc1 = bias_variable([200])
# [n_samples, 7, 7, 32] ->> [n_samples, 7x7x32]
 
#這裡將第2層max_pool 過後的神經元 全部攤平
h_pool4_flat = tf.reshape(h_pool4, [-1, 7*7*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
#倒數第二層為1024個神經元 最後一層為10個神經元 採用softmax當成最後一層的激活函數
W_fc2 = weight_variable([200, 15])
b_fc2 = bias_variable([15])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 

 
# 定義loss function 以及 優化函數
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))# loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#定義Sess 以及初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
#開始訓練，dropout 0.5代表隨機隱藏掉一半神經元的資訊
#科學家們發現這樣可以有效的減少overfitting
#有關dropout的相關資訊可以參考這篇


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

for i in range(1000):
    sess.run(iter.initializer)
    for j in range(int(len(label_list)/100)):
        img , label = sess.run(el)
        sess.run(train_step, feed_dict={xs:img, ys:label, keep_prob: 0.5})
    print(compute_accuracy(img_test, label_test))
