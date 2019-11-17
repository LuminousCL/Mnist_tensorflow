#####基于mnist手写数字数据集的数字识别小程序#####
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
mnist = input_data.read_data_sets('/temp/', one_hot=True)

#设置
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#预测值
y = tf.nn.softmax(tf.matmul(x,W)+b)
#真值
y_ = tf.placeholder(tf.float32,[None,10])
#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#使用优化器
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化变量
init = tf.initialize_all_variables()
#创建对话
sess = tf.Session()
sess.run(init)


for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))