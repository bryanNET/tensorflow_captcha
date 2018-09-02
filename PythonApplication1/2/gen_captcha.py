#coding:utf-8 
#import tensorflow as tf
#w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
#w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#x = tf.constant([[0.7,0.9]])
#a = tf.matmul(x,w1)
#y = tf.matmul(a,w2)
#sess = tf.Session()
#init_op = tf.initialize_all_variables()
#sess.run(init_op)
#print(sess.run(y))
#sess.close()

 

#a = tf.random_normal((100, 100))
#b = tf.random_normal((100, 500))
#c = tf.matmul(a, b)
#sess = tf.InteractiveSession()

#print(sess.run(c))

##print("hello,world")

#coding=utf-8
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
import random
from captcha.image import ImageCaptcha # pip install captcha 
from PIL import Image
from io import BytesIO

 


from PIL import Image, ImageDraw, ImageFont, ImageFilter


# 验证码中的字符, 就不用汉字了
 
#number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#      'v', 'w', 'x', 'y', 'z']
 
#ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#      'V', 'W', 'X', 'Y', 'Z']

'''
,'1','2','3','4'
'''
number=['0','1']
alphabet =[]
ALPHABET =[]

 
# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
  captcha_text = []
  for i in range(captcha_size):
    c = random.choice(char_set)
    captcha_text.append(c)
  return captcha_text
 
 
# 生成字符对应的验证码
def gen_captcha_text_and_image1():
  while(1):

      
    image = ImageCaptcha()

    # 创建Font对象:
    #font = ImageFont.truetype('Arial.ttf', 36) 
    # 模糊:
    # image = image.filter(ImageFilter.BLUR)

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
 

    #captcha = image.generate(captcha_text)
    #captcha = downimage(captcha_text)
    #image.write(captcha_text, captcha_text + '.jpg') # 写到文件
 
    #captcha_image = Image.open(captcha)   
    #captcha_image.show()

     
    captcha_image = downimage(captcha_text)
    #captcha_image.show()
    captcha_image = np.array(captcha_image)
    
     #if captcha_image.shape==(80,200,4):
    if captcha_image.shape==(60,160,4):
      break
 
  return captcha_text, captcha_image
 
#取html 验证码
def downimage(code):
    # 构建session
    sess = requests.Session()
    url="http://localhost:8200/CaptchaWeb.aspx?code="+code
    #response = requests.get(url)
    #image = Image.open(BytesIO(response.content))
    #image = resize_image(image, 60, 160)

    image = sess.get(url).content
    captcha_image = Image.open(BytesIO(image))
    captcha_image = np.array(captcha_image)  
    image = resize_image(captcha_image, 60, 160) 
    return image

    # 建立请求头
    #headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36", "Connection": "keep-alive"}
    # 这个url是联合航空公司验证码，根据访问时间戳返回图片https://account.flycua.com/sso/chineseVerifyCode.images
   
  

    # 获取响应图片内容
    #image=sess.get(url,headers=headers).content
    # 保存到本地
    #with open(str(i)+"image.jpg","wb") as f:
    #    f.write(image)
    #return image 

    #res = requests.get(url,headers=headers)
    #byte_stream = io.BytesIO(res.content) # 把请求到的数据转换为Bytes字节流(这样解释不知道对不对，可以参照[廖雪峰](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431918785710e86a1a120ce04925bae155012c7fc71e000)的教程看一下)
 
    #roiImg = Image.open(byte_stream)  # Image打开二进制流Byte字节流数据
    #return roiImg
    #imgByteArr = io.BytesIO()   # 创建一个空的Bytes对象 
    #roiImg.save(imgByteArr, format='PNG') # PNG就是图片格式，我试过换成JPG/jpg都不行 
    #imgByteArr = imgByteArr.getvalue()  # 这个就是保存的二进制流

#按照指定图像大小调整尺寸
def resize_image(image, height, width):
     top, bottom, left, right = (0, 0, 0, 0)

     #获取图像尺寸
     #h, w, _ = image.shape

     ##对于长宽不相等的图片，找到最长的一边
     #longest_edge = max(h, w)    

     ##计算短边需要增加多上像素宽度使其与长边等长
     #if h < longest_edge:
     #    dh = longest_edge - h
     #    top = dh // 2
     #    bottom = dh - top
     #elif w < longest_edge:
     #    dw = longest_edge - w
     #    left = dw // 2
     #    right = dw - left
     #else:
     #    pass 

     #RGB颜色
     BLACK = [0, 0, 0]

     #给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
     constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)

     #调整图像大小并返回     
     return cv2.resize(constant, (width,height))

 
#if __name__ == '__main__':
#  # 测试
#  text, image = gen_captcha_text_and_image()
#  print(image)
#  gray = np.mean(image, -1)
#  print(gray)
 
#  print(image.shape)
#  print(gray.shape)
#  f = plt.figure()
#  ax = f.add_subplot(111)
#  ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
#  plt.imshow(image)
 
#  plt.show()
