import tensorflow as tf  
from tensorflow.python.framework import graph_util  
from tensorflow.python.platform import gfile  
  
if __name__ == "__main__":  
    a = tf.Variable(tf.constant(5.,shape=[1]),name="a")  
    b = tf.Variable(tf.constant(6.,shape=[1]),name="b")  
    c = a + b  
    init = tf.initialize_all_variables()  
    sess = tf.Session()  
    sess.run(init)  
    #导出当前计算图的GraphDef部分  
    graph_def = tf.get_default_graph().as_graph_def()  
    #保存指定的节点，并将节点值保存为常数  
    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])  
    #将计算图写入到模型文件中  
    model_f = tf.gfile.GFile("model.pb","wb")  
    model_f.write(output_graph_def.SerializeToString())  


#保存模型代码 .pb

import tensorflow as tf
from tensorflow.python.framework import graph_util
var1 = tf.Variable(1.0, dtype=tf.float32, name='v1')
var2 = tf.Variable(2.0, dtype=tf.float32, name='v2')
var3 = tf.Variable(2.0, dtype=tf.float32, name='v3')
x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
x2 = tf.placeholder(dtype=tf.float32, shape=None, name='x2')
addop = tf.add(x, x2, name='add')
addop2 = tf.add(var1, var2, name='add2')
addop3 = tf.add(var3, var2, name='add3')
initop = tf.global_variables_initializer()
model_path = './Test/model.pb'
with tf.Session() as sess:
    sess.run(initop)
    print(sess.run(addop, feed_dict={x: 12, x2: 23}))
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['add', 'add2', 'add3'])
    # 将计算图写入到模型文件中
    model_f = tf.gfile.FastGFile(model_path, mode="wb")
    model_f.write(output_graph_def.SerializeToString())



#读取模型代码
import tensorflow as tf
with tf.Session() as sess:
    model_f = tf.gfile.FastGFile("./Test/model.pb", mode='rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model_f.read())
    c = tf.import_graph_def(graph_def, return_elements=["add2:0"])
    c2 = tf.import_graph_def(graph_def, return_elements=["add3:0"])
    x, x2, c3 = tf.import_graph_def(graph_def, return_elements=["x:0", "x2:0", "add:0"])

    print(sess.run(c))
    print(sess.run(c2))
    print(sess.run(c3, feed_dict={x: 23, x2: 2}))





























#import os
#import shutil
#import random
#import time
##captcha是用于生成验证码图片的库，可以 pip install captcha 来安装它
#from captcha.image import ImageCaptcha
 
##用于生成验证码的字符集
#CHAR_SET = ['0','1','2','3','4','5','6','7','8','9']
##字符集的长度
#CHAR_SET_LEN = 10
##验证码的长度，每个验证码由4个数字组成
#CAPTCHA_LEN = 4
 
##验证码图片的存放路径
#CAPTCHA_IMAGE_PATH = 'C:/Users/bryan/Desktop/AI/img/'
##用于模型测试的验证码图片的存放路径，它里面的验证码图片作为测试集
#TEST_IMAGE_PATH = 'C:/Users/bryan/Desktop/AI/test/'
##用于模型测试的验证码图片的个数，从生成的验证码图片中取出来放入测试集中
#TEST_IMAGE_NUMBER = 50
 
##生成验证码图片，4位的十进制数字可以有10000种验证码
#def generate_captcha_image(charSet = CHAR_SET, charSetLen=CHAR_SET_LEN, captchaImgPath=CAPTCHA_IMAGE_PATH):   
#    k  = 0
#    total = 1
#    for i in range(CAPTCHA_LEN):
#        total *= charSetLen
        
#    for i in range(charSetLen):
#        for j in range(charSetLen):
#            for m in range(charSetLen):
#                for n in range(charSetLen):
#                    captcha_text = charSet[i] + charSet[j] + charSet[m] + charSet[n]
#                    image = ImageCaptcha()
#                    image.write(captcha_text, captchaImgPath + captcha_text + '.jpg')
#                    k += 1
#                    sys.stdout.write("\rCreating %d/%d" % (k, total))
#                    sys.stdout.flush()
                    
##从验证码的图片集中取出一部分作为测试集，这些图片不参加训练，只用于模型的测试                    
#def prepare_test_set():
#    fileNameList = []    
#    for filePath in os.listdir(CAPTCHA_IMAGE_PATH):
#        captcha_name = filePath.split('/')[-1]
#        fileNameList.append(captcha_name)
#    random.seed(time.time())
#    random.shuffle(fileNameList) 
#    for i in range(TEST_IMAGE_NUMBER):
#        name = fileNameList[i]
#        shutil.move(CAPTCHA_IMAGE_PATH + name, TEST_IMAGE_PATH + name)
                        
#if __name__ == '__main__':
#    generate_captcha_image(CHAR_SET, CHAR_SET_LEN, CAPTCHA_IMAGE_PATH)
#    prepare_test_set()
#    sys.stdout.write("\nFinished")
#    sys.stdout.flush()  