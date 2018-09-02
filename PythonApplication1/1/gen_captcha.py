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
def gen_captcha_text_and_image():
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


#按照指定图像大小调整尺寸
def resize_image(image, height, width):
     top, bottom, left, right = (0, 0, 0, 0)

   

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
