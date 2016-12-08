# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:26:22 2016

@author: Bertrand
"""

from keras import backend as K
K.set_image_dim_ordering('th')
import os
from PIL import Image
import numpy as np
from keras.models import load_model

#%%
'''
set variables
'''
img_rows, img_cols = 48,48
max_image = 50
rgb_max = 255
batch_size = 128

#%%
'''
load images and preprocess
'''

img = []
size = (img_rows, img_cols)
data = np.empty((max_image,1,img_rows, img_cols),dtype="float32")
num = 0

def load_data():
    imgs = os.listdir("./image")
    num = len(imgs)
    for i in range(num):     
        img = Image.open("./image/"+imgs[i])
        img = img.convert("L")
        img = img.resize(size)
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
    return data,num
data,num = load_data()

img_ndarray = np.asarray(data, dtype='float64')/rgb_max
faces=np.empty((num,2304))
for i in range(num):
    faces[i]=np.ndarray.flatten(img_ndarray [i,0:img_rows,0:img_cols])
print (faces[0])

#%%
'''
predict images and print results
''' 
predict_data=np.empty((num, img_rows * img_cols))
for i in range(num):
    predict_data[i] = faces[i]
Faces = predict_data.reshape(predict_data.shape[0],1,img_rows,img_cols)

model = load_model("model.h5")
result = model.predict_classes(Faces, batch_size=batch_size, verbose=1)

number=0
print (result)
for i in range(result.size):
    print("The expression of image " + str(number) + " is:")
    if(result[i] == [0]): print("ANGRY!(◣_◢)")
    elif(result[i] == [1]): print("DISGUST! ewww~~")
    elif(result[i] == [2]): print("FEAR! Σ( ° △ °|||)︴")
    elif(result[i] == [3]): print("HAPPY!~(￣▽￣)~* ")
    elif(result[i] == [4]): print("SAD! (╥_╥)")
    elif(result[i] == [5]): print("SURPRISE!（⊙o⊙）")
    else: print("NEUTRAL :|")
    number+=1
