# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 下午8:18
# @Author  : xuef
# @FileName: train_captcha.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/weixin_42118777/article
"""
function:识别captcha验证码技术
生成验证码函数
"""
from captcha.image import ImageCaptcha  # pip install captcha_recognize
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# 验证码一般都无视大小写；验证码长度4个字符 这里为节省演示时间 将alphabet ALPHABET注释掉
# def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)  # 用字符['m', 'i', 'R', 'd']转化为miRd

    captcha = image.generate(captcha_text)   # 将captcha_text转化为图片
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
    captcha_image = Image.open(captcha)
    # captcha_image.show() 测试是否成功生成验证码
    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image


# if __name__ == '__main__':
#     # 测试
#     text, image = gen_captcha_text_and_image()
#
#     f = plt.figure()
#     ax = f.add_subplot(111)
#     ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
#     plt.imshow(image)