# coding:utf-8
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from gen_captcha import resize_image


def downimage(i):
    # 构建session
    sess = requests.Session()
    # 建立请求头
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.75 Safari/537.36",
             "Connection": "keep-alive"}
    # 这个url是验证码， 
    url="http://localhost:8010/CaptchaWeb.aspx?code=asdf"

    # 获取响应图片内容
    image=sess.get(url,headers=headers).content
    captcha_image = Image.open(BytesIO(image))
    captcha_image = np.array(captcha_image)  

    image = resize_image(captcha_image, 60, 160)
    # 保存到本地
    cv2.imwrite(str(i)+"image.jpg", image) 

if __name__=="__main__":
    # 获取10张图片
    for i in range(5):
        downimage(i)
