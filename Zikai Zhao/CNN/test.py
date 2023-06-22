import numpy as np
from PIL import Image
from skimage.io import imread

image=Image.open('1.jpg')
s=imread('1.jpg')
print(s)
# print(image)
image=np.copy(image) # 这一句
# s=np.asarray("1.jpg")

# print(image)