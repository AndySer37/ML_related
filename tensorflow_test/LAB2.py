# Copyright 2017 Google, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import os.path
import shutil
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LOGDIR = "/tmp/demo3/"
LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")
### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)




# Define a simple convolutional layer
def conv_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    #tf.summary.histogram("weights", w)
    #tf.summary.histogram("biases", b)
    #tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# And a fully connected layer
def fc_layer(input, size_in, size_out):
  
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    act = tf.matmul(input, w) + b
    #tf.summary.histogram("weights", w)
    #tf.summary.histogram("biases", b)
    #tf.summary.histogram("activations", act)
    return act


def mnist_model(learning_rate, hparam):
  tf.reset_default_graph()
  sess = tf.Session()
  
  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784])
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  #tf.summary.image('input', x_image, 3)
  y = tf.placeholder(tf.float32, shape=[None, 10])

  #build your model
  conv1 = conv_layer(x_image, 1, 32)
  conv_out = conv_layer(conv1, 32, 64)
  flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])
  fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
  relu = tf.nn.relu(fc1)
  tf.summary.histogram("fc1/relu", relu)
  logits = fc_layer(relu, 1024, 10)
  

  #Loss&trainging
  #with tf.name_scope("xent"):
  xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
  #tf.summary.scalar("xent", xent)
  
  #with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)
  
  #with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  #tf.summary.scalar("accuracy", accuracy)
    
  
  
  #summ = tf.summary.merge_all()
  #writer = tf.summary.FileWriter(LOGDIR+hparam)
  #writer.add_graph(sess.graph)

  
  sess.run(tf.global_variables_initializer())
  for i in range(1000):
    batch = mnist.train.next_batch(100)
    # Occasionally report accuracy  
    #if i % 5 == 0:
     # [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
      #writer.add_summary(s, i)
    if i % 500 == 0:
      [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})  
      print("step %d, training accuracy %g" % (i, train_accuracy))

    # Run the training step
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
    


def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-4]:
    conv_param="conv=2"
    fc_param= "fc=2"
    # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
    hparam = "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)
    print('Starting run for %s' % hparam)

    # Actually run with the new settings
    mnist_model(learning_rate, hparam)
  print('Done training!')
  print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
  print('Running on mac? If you want to get rid of the dialogue asking to give '
        'network permissions to TensorBoard, you can provide this flag: '
        '--host=localhost')

if __name__ == '__main__':
  start = time.clock()
  main()
  end= time.clock()
  print('tun time: %f s' %(end-start))
