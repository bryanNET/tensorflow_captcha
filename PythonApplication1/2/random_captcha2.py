from captcha.image import ImageCaptcha
from gen_captcha import gen_captcha_text_and_image1
import numpy as  np
import matplotlib.pyplot as  plt
from  PIL import Image
import random
import tensorflow as tf

#number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#            'v', 'w', 'x', 'y', 'z']
#Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#            'V', 'W', 'X', 'Y', 'Z']
number = ['0', '1']
alphabet = []
Alphabet = []
char_set = number + alphabet + Alphabet

##图片高
IMAGE_HEIGHT = 60
##图片宽
IMAGE_WIDTH = 160
##验证码长度
MAX_CAPTCHA = 4
##验证码选择空间
CHAR_SET_LEN = len(char_set)
##提前定义变量空间
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  ##节点保留率


##生成n位验证码字符 这里n=4
def random_captcha_text(char_set=char_set, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


##使用ImageCaptcha库生成验证码
def gen_captcha_text_and_image():
    #image = ImageCaptcha()
    #captcha_text = random_captcha_text()
    #captcha_text = ''.join(captcha_text)
    #captcha = image.generate(captcha_text)
    #captcha_image = Image.open(captcha)
    #captcha_image = np.array(captcha_image)
    captcha_text, captcha_image = gen_captcha_text_and_image1() 
    return captcha_text, captcha_image

def gen_captcha_text_and_image0():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


##彩色图转化为灰度图
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


##获取字符在 字符域中下标
def getPos(char_set=char_set, char=None):
    return char_set.index(char)


##验证码字符转换为长向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    """
    def char2pos(c):  
        if c =='_':  
            k = 62  
            return k  
        k = ord(c)-48  
        if k > 9:  
            k = ord(c) - 55  
            if k > 35:  
                k = ord(c) - 61  
                if k > 61:  
                    raise ValueError('No Map')   
        return k  
    """
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + getPos(char=c)
        vector[idx] = 1
    return vector


##获得1组验证码数据
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while 1:
            text, image = gen_captcha_text_and_image()
            #if image.shape == (60, 160, 3):
            if image.shape == (60, 160, 4):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


##卷积层 附relu  max_pool drop操作
def conn_layer(w_alpha=0.01, b_alpha=0.1, _keep_prob=0.7, input=None, last_size=None, cur_size=None):
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, last_size, cur_size]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([cur_size]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob=_keep_prob)
    return conv1


##对卷积层到全链接层的数据进行变换
def _get_conn_last_size(input):
    shape = input.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    input = tf.reshape(input, [-1, dim])
    return input, dim


##全链接层
def _fc_layer(w_alpha=0.01, b_alpha=0.1, input=None, last_size=None, cur_size=None):
    w_d = tf.Variable(w_alpha * tf.random_normal([last_size, cur_size]))
    b_d = tf.Variable(b_alpha * tf.random_normal([cur_size]))
    fc = tf.nn.bias_add(tf.matmul(input, w_d), b_d)
    return fc


##构建前向传播网络
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    conv1 = conn_layer(input=x, last_size=1, cur_size=32)
    conv2 = conn_layer(input=conv1, last_size=32, cur_size=64)
    conn3 = conn_layer(input=conv2, last_size=64, cur_size=64)

    input, dim = _get_conn_last_size(conn3)

    fc_layer1 = _fc_layer(input=input, last_size=dim, cur_size=1024)
    fc_layer1 = tf.nn.relu(fc_layer1)
    fc_layer1 = tf.nn.dropout(fc_layer1, keep_prob)

    fc_out = _fc_layer(input=fc_layer1, last_size=1024, cur_size=MAX_CAPTCHA * CHAR_SET_LEN)
    return fc_out

#构建卷积神经网络并训练   定义CNN
def crack_captcha_cnn2(w_alpha=0.01, b_alpha=0.1):
    #数据变换 
    # 把 X reshape 成 IMAGE_HEIGHT*IMAGE_WIDTH*1的格式,输入的是灰度图片，所有通道数是1;
    # shape 里的-1表示数量不定，根据实际情况获取，这里为每轮迭代输入的图像数量（batchsize）的大小;
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
 
    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)
 
    # 3 conv layer
    #第一层卷积
    # shape[3, 3, 1, 32]里前两个参数表示卷积核尺寸大小，即patch;
    # 第三个参数是图像通道数，第四个参数是该层卷积核的数量，有多少个卷积核就会输出多少个卷积特征图像
    
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
  
    # 每个卷积核都配置一个偏置量，该层有多少个输出，就应该配置多少个偏置量
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # 图片和卷积核卷积，并加上偏执量，卷积结果28x28x32
    # tf.nn.conv2d() 函数实现卷积操作
    # tf.nn.conv2d()中的padding用于设置卷积操作对边缘像素的处理方式，在tf中有VALID和SAME两种模式
    # padding='SAME'会对图像边缘补0,完成图像上所有像素（特别是边缘象素）的卷积操作
    # padding='VALID'会直接丢弃掉图像边缘上不够卷积的像素
    # strides：卷积时在图像每一维的步长，是一个一维的向量，长度4，并且strides[0]=strides[3]=1
    # tf.nn.bias_add() 函数的作用是将偏置项b_c1加到卷积结果value上去;
    # 注意这里的偏置项b_c1必须是一维的，并且数量一定要与卷积结果value最后一维数量相同
    # tf.nn.relu() 函数是relu激活函数，实现输出结果的非线性转换，即features=max(features, 0)，输出tensor的形状和输入一致
   
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))

    # tf.nn.max_pool()函数实现最大池化操作，进一步提取图像的抽象特征，并且降低特征维度
    # ksize=[1, 2, 2, 1]定义最大池化操作的核尺寸为2*2, 池化结果14x14x32 卷积结果乘以池化卷积核    
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # tf.nn.dropout是tf里为了防止或减轻过拟合而使用的函数，一般用在全连接层;
    # Dropout机制就是在不同的训练过程中根据一定概率（大小可以设置，一般情况下训练推荐0.5）随机扔掉（屏蔽）一部分神经元，
    # 不参与本次神经网络迭代的计算（优化）过程，权重保留但不做更新;
    # tf.nn.dropout()中 keep_prob用于设置概率，需要是一个占位变量，在执行的时候具体给定数值    
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层卷积（图像尺寸不变、特征×32）、池化（图像尺寸缩小一半，特征不变）之后;
    # 输出大小为 30*80*32


    #第二层卷积  
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层后输出大小为 30*80*32
    # 经过神经网络第二层运算后输出为 16*40*64 (30*80的图像经过2*2的卷积核池化，padding为SAME，输出维度是16*40)


    #第三层卷积  
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # 原图像HEIGHT = 60 WIDTH = 160，经过神经网络第一层后输出大小为 30*80*32 经过第二层后输出为 16*40*64
    # 经过神经网络第二层运算后输出为 16*40*64 ; 经过第三层输出为 8*20*64，这个参数很重要，决定量后边全连接层的维度


 
    #全链接层  
    #每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍  
    #完全连接层 Fully connected layer
    # 二维张量，第一个参数8*20*64的patch，这个参数由最后一层卷积层的输出决定，第二个参数代表卷积个数共1024个，即输出为1024个特征
    
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]))
  
    # 偏置项为1维，个数跟卷积核个数保持一致
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    # w_d.get_shape()作用是把张量w_d的形状转换为元组tuple的形式，w_d.get_shape().as_list()是把w_d转为元组再转为list形式
    # w_d 的 形状是[ 8 * 20 * 64, 1024]，w_d.get_shape().as_list()结果为 8*20*64=10240 ;
    # 所以tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])的作用是把最后一层隐藏层的输出转换成一维的形式
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    # tf.matmul(dense, w_d)函数是矩阵相乘，输出维度是 -1*1024
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
    # 经过全连接层之后，输出为 一维，1024个向量
 
    #输出层  
  
    # w_out定义成一个形状为 [1024, 8 * 10] = [1024, 80]
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    # out 的输出为 8*10 的向量， 8代表识别结果的位数，10是每一位上可能的结果（0到9）
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    # 输出神经网络在当前参数下的预测值
    return out
 

##反向传播
def back_propagation():
    output = crack_captcha_cnn()
    ##学习率 reduce_mean 函数的作用是求平均值
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, Y))
    #sigmoid_cross_entropy_with_logits 针对分类问题 计算具有权重的sigmoid交叉熵 [tensorflow损失函数系列]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output,logits=Y))
    optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.arg_max(predict, 2)
    max_idx_l = tf.arg_max(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(max_idx_p, max_idx_l), tf.float32))
    return loss, optm, accuracy


##初次运行训练模型
def train_first():
    import time
    loss, optm, accuracy = back_propagation()

    saver = tf.train.Saver()
    with tf.Session() as  sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while 1:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optm, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75}) 
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),step, loss_)
            if step % 50 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("------------第{}次的准确率为 {} 损失{}".format(step, acc, loss_))# print(step, acc, loss_)
                if acc > 0.71:  ##准确率大于0.80保存模型 可自行调整
                    saver.save(sess, 'models/crack_capcha.model', global_step=step)
                    break
            step += 1


##加载现有模型 继续进行训练
def train_continue(step):
    loss, optm, accuracy = back_propagation()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        path = "models/crack_capcha.model-" + str(step)
        saver.restore(sess, path)
        ##36300 36300 0.9325 0.0147698
        while 1:
            batch_x, batch_y = get_next_batch(100)
            _, loss_ = sess.run([optm, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            if step % 50 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(" {}步 : {} 损失{}".format(step, acc, loss_))# print(step, acc, loss_)
                if acc >= 0.925:
                    saver.save(sess, 'models/crack_capcha.model', global_step=step)
                if acc >= 0.95:
                    saver.save(sess, 'models/crack_capcha.model', global_step=step)
                    break
            step += 1


##测试训练模型
##测试训练模型
def crack_captcha(captcha_image, step):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        path = 'models/crack_capcha.model-' + str(step)
        saver.restore(sess, path)

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text

def crack_captcha2(captcha_image, step):
    output = crack_captcha_cnn2() 
    saver = tf.train.Saver()
    with tf.Session() as sess: 
        path = 'models1/crack_capcha.model-100'
        saver.restore(sess, path)

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2) 
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1}) 
        text = text_list[0].tolist()
        return text


if __name__ == '__main__':
    ##训练和测试开关
    train = 1
    if train:# if >0
        ##train_continue(36300)
        train_first()
    else: 
        text, image = gen_captcha_text_and_image1() 
        if image.shape == (60, 160, 4): 
            ##(80,200,4):#(60, 160, 3):
            #text, image = gen_captcha_text_and_image()

            f = plt.figure()
            ax = f.add_subplot(111)
            ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
            plt.imshow(image)
            plt.show()

            image = convert2gray(image)
            image = image.flatten() / 255

            #predict_text = crack_captcha2(image, 5)
            predict_text = crack_captcha(image, 0)
            print("正确: {}  预测: {}".format(text, [char_set[char] for i, char in enumerate(predict_text)]))
