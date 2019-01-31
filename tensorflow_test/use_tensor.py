import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import random
from PIL import Image
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

xs = tf.placeholder(tf.float32, shape=(None,101,101,3),name="xs") 
ys = tf.placeholder(tf.float32, shape=(None,15),name="ys")

keep_prob = tf.placeholder(tf.float32)

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
#tf.reset_default_graph()

with tf.Session() as sess:
    with gfile.FastGFile('./'+'model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name="")
    #new_saver = tf.train.import_meta_graph('model_name.meta')
    #new_saver.restore(sess, 'model_name')
    #train_step = tf.get_collection("train_step")[0]
    sess.run(tf.global_variables_initializer())
    op = sess.graph.get_tensor_by_name('out:0')
    input_x = sess.graph.get_tensor_by_name('x:0')
    #input_y = sess.graph.get_tensor_by_name('y:0')
    keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
    print sess.run(op, feed_dict={input_x:img_test,  keep_prob:0.8})
    
    
tf.reset_default_graph()
