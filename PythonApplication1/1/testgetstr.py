#from train import crack_captcha_cnn
#from train import convert2gray
#from train import vec2text
#from gen_captcha import number
#from gen_captcha import gen_captcha_text_and_image
#from gen_captcha import alphabet  
#from PIL import Image
 
import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time
 
MODEL_SAVE_PATH = 'C:/Users/bryan/Desktop/AI/PythonApplication1/PythonApplication1/'  

number = ['0', '1']
# 图像大小, '2', '3', '4', '5', '6', '7', '8', '9'
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
char_set = number
CHAR_SET_LEN = len(char_set)
 
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout
 
# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
 
    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
 
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
 
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
 
    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)
 
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out
 
# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
 
def predict_captcha(captcha_image):
    output = crack_captcha_cnn()
 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
 
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
 
        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)

#单张图片预测MODEL_SAVE_PATH+
image = np.float32(cv2.imread('./0111.png', 0))
text = '0111'
image = image.flatten() / 255
predict_text = predict_captcha(image)
print("正确: {0}  预测: {1}".format(text, predict_text))


#def testdownimage():
#       #单张图片预测MODEL_SAVE_PATH+
#        image = np.float32(cv2.imread('./0111.png', 0))
#        text = '0111'
#        image = image.flatten() / 255
#        predict_text = predict_captcha(image)
#        print("正确: {0}  预测: {1}".format(text, predict_text))

#CAPTCHA_LEN = 4  
  
#

#def testdownimage():
#    output = crack_captcha_cnn()
#    #saver = tf.train.Saver()
#     #加载graph  
#    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+"crack_capcha.model-100.meta")  
#    sess = tf.Session()
#    saver.restore(sess, tf.train.latest_checkpoint('.'))
#    with tf.Session() as sess:  
#       saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH)  

#    while(1): 
#      text, image = gen_captcha_text_and_image()
#      image = convert2gray(image)
#      image = image.flatten() / 255
 
#      predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
#      text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
#      predict_text = text_list[0].tolist()
 
#      vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
#      i = 0
#      for t in predict_text:
#        vector[i * 63 + t] = 1
#        i += 1
#        # break
#      print("正确: {} 预测: {}".format(text, vec2text(vector)))



#if __name__=="__main__":
#      testdownimage()



  

#TEST_IMAGE_PATH = 'E:/Tensorflow/captcha/test/'  
  
#def get_image_data_and_name(fileName, filePath=TEST_IMAGE_PATH):  
#    pathName = os.path.join(filePath, fileName)  
#    img = Image.open(pathName)  
#    #转为灰度图  
#    img = img.convert("L")         
#    image_array = np.array(img)      
#    image_data = image_array.flatten()/255  
#    image_name = fileName[0:CAPTCHA_LEN]  
#    return image_data, image_name  
  
#def digitalStr2Array(digitalStr):  
#    digitalList = []  
#    for c in digitalStr:  
#        digitalList.append(ord(c) - ord('0'))  
#    return np.array(digitalList)  
  
#def model_test():  
#    #加载graph  
#    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+"crack_capcha.model-100.meta")  
#    graph = tf.get_default_graph()  
#    #从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)  
#    input_holder = graph.get_tensor_by_name("data-input:0")  
#    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")  
#    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")  
#    with tf.Session() as sess:  
#        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))  
#        count = 0 
#        for i in range(5):
#            #img_data, img_name = get_image_data_and_name(fileName, TEST_IMAGE_PATH)  
#            predict = sess.run(predict_max_idx, feed_dict={input_holder:[img_data], keep_prob_holder : 1.0})              
           
#            text, image = gen_captcha_text_and_image()
#            #  image = convert2gray(image)
##  image = image.flatten() / 255
 
##  predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
##  text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
##  predict_text = text_list[0].tolist()
 
#            IMAGE_HEIGHT = 60
#            IMAGE_WIDTH = 160
#            MAX_CAPTCHA = len(text)

#            plt.imshow(img)  
#            plt.axis('off')  
#            plt.show()  
#            image = convert2gray(image)

#            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

#            print("正确: {} 预测: {}".format(text, vec2text(vector)))
#            #filePathName = TEST_IMAGE_PATH + fileName  
#            #print(filePathName)  
#            #img = Image.open(filePathName)  
#            #plt.imshow(img)  
#            #plt.axis('off')  
#            #plt.show()  
#            #predictValue = np.squeeze(predict)  
#            #rightValue = digitalStr2Array(img_name)  
#            #if np.array_equal(predictValue, rightValue):  
#            #    result = '正确'  
#            #    count += 1  
#            #else:   
#            #    result = '错误'              
#            #print('实际值：{}， 预测值：{}，测试结果：{}'.format(rightValue, predictValue, result))  
#            #print('\n')  
          
              
#       # print('正确率：%.2f%%(%d/%d)' % (count*100/totalNumber, count, totalNumber))  


#if __name__ == '__main__': 
#  model_test()

