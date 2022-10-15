# Aman Kumar
# 2020CSB1153
# Task 2


# importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import data 
from skimage.color import *
from skimage import feature
from skimage.filters import laplace
from skimage.transform import resize
from skimage.io import imread

# Function to calculate convolution
def convolution(img , kernel):
    img_r , img_c = img.shape
    krnl_r , krnl_c = kernel.shape

    op = np.zeros((img_r,img_c))

    ht = int((krnl_r-1)/2)
    wd = int((krnl_c-1)/2)

    pad_img = np.zeros((img_r + 2*ht , img_c + 2*wd))

    for i in range(img_r):
        for j in range(img_c):
            pad_img[i+ht][j+wd] = img[i][j]

    for i in range(img_r):
        for j in range(img_c):
            prod = kernel
            arr = pad_img[i:i+krnl_r,j:j+krnl_c]
            prod = prod*arr
            for i1 in range(krnl_r):
                for j1 in range(krnl_c):
                    op[i][j] += prod[i1][j1]

    return op


# Function to calculate Variance of Laplacian
def LaplacianVariance(inputImg):
    opr = np.array([[0,1,0], [1, -4, 1], [0, 1, 0]])
    res = laplace(inputImg,ksize=3)

    var = np.mean((res - res.mean())**2)

    return var


# Sigmoid function calculator
def sigmoid_function(val):
    x = (150.0-val)/255.0
    res = 1 + np.e**(x)
    res = 1.0/res

    return 1-res

# BlurOrNot function to check if the given image is blurred or not
def BlurOrNot(img):
    var = LaplacianVariance(img)
    var = var*255*255
    res = sigmoid_function(var)
    if(res < 0.5):
        print("The Image is Not Blurred")
    else:
        print("The Image is Blurred")
    print("Probability of Blurred : ",res)


# Main Function
if __name__ == '__main__':
    inputImg = imread('image/img2.jpg')
    if len(inputImg.shape)==3:
        inputImg = rgb2gray(inputImg)
    BlurOrNot(inputImg)
    
    