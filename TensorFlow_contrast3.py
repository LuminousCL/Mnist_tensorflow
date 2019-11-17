#####TensorFlow对mnist数据集的操作#####
from tensorflow.examples.tutorials.mnist import input_data
# 第一次运行会自动下载到代码所在的路径下

mnist = input_data.read_data_sets('location', one_hot=True)
# location 是保存的文件夹的名称

# 打印 mnist 的一些信息
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print("type of 'mnist is %s'" % (type(mnist)))
print("number of train data is %d" % mnist.train.num_examples)
print("number of test data is %d" % mnist.test.num_examples)

# 将所有的数据加载为这样的四个数组 方便之后的使用
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print("Type of training is %s" % (type(trainimg)))
print("Type of trainlabel is %s" % (type(trainlabel)))
print("Type of testing is %s" % (type(testimg)))
print("Type of testing is %s" % (type(testlabel)))

# 可视化数据保存的图片
import numpy as np
import matplotlib.pyplot as plt


nsmaple = 4
randidx = np.random.randint(trainimg.shape[0], size=nsmaple)

for i in randidx:
    curr_img = np.reshape(trainimg[i,:], (28, 28))  # 数据中保存的是 1*784 先reshape 成 28*28
    curr_label = np.argmax(trainlabel[i, :])
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.show()


