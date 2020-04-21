# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#image pyramid 解決物體尺度變化問題

def gaussian(img):
    '''
    g_filter = np.array([[1,2,1],
                         [2,4,2],
                         [1,2,1]])/16
    '''
    g_filter = np.array([[  1,  4,  6,  4,  1],
                         [  4, 16, 24, 16,  4],
                         [  6, 24, 36, 24,  6],
                         [  4, 16, 24, 16,  4],
                         [  1,  4,  6,  4,  1]])/256
    
    height = img.shape[0]
    width = img.shape[1]
    num = g_filter.shape[0]//2
    img_padding = np.pad(img,((num,num),(num,num)),'constant',constant_values=0)
    result = np.zeros([height, width], dtype = int)
    
    for i in range(height):
        for j in range(width):
            result[i,j] = int(np.sum(img_padding[i:i+(2*num)+1, j:j+(2*num)+1]*g_filter))
            
    return result

def subsampling(img):
    height = img.shape[0]
    width = img.shape[1]
    result = np.zeros(((height+1)//2, (width+1)//2),dtype = int)
    for i in range((height+1)//2):
        for j in range((width+1)//2):
            result[i,j] = int(np.sum(img[2*i:2*i+2,2*j:2*j+2])/4)
    return result

def magnitude_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def resize(src, new_sizeh,new_sizew):
    dst_h, dst_w = new_sizeh, new_sizew # 目标图像宽高
    src_h, src_w = src.shape[:2] # 源图像宽高
    if src_h == dst_h and src_w == dst_w:
        return src.copy()
    scale_x = float(src_w) / dst_w # x缩放比例
    scale_y = float(src_h) / dst_h # y缩放比例

    # 遍历目标图像，插值
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
    for dst_y in range(dst_h): # 对height循环
        for dst_x in range(dst_w): # 对width循环
            # 目标在源上的坐标
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            # 计算在源图上四个近邻点的位置
            src_x_0 = int(np.floor(src_x))
            src_y_0 = int(np.floor(src_y))
            src_x_1 = min(src_x_0 + 1, src_w - 1)
            src_y_1 = min(src_y_0 + 1, src_h - 1)
            # 双线性插值
            value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0] + (src_x - src_x_0) * src[src_y_0, src_x_1]
            value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0] + (src_x - src_x_0) * src[src_y_1, src_x_1]
            dst[dst_y, dst_x] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
    return dst
    

def gaussian_pyramid(img,level):
    pyramid_images = []
    spectrum = []
    temp = img
    for i in range(level):
        if i == 0:
            pyramid_images.append(temp)
        else:
            g = gaussian(temp)
            sub = subsampling(g)
            pyramid_images.append(sub)
            temp = sub
        sp = magnitude_spectrum(temp)
        spectrum.append(sp)
    return pyramid_images, spectrum
'''
def laplacian_pyramid(gaussian_images,level):
    pyramid_images = []
    for i in range(level):
        if i == (level-1):
            pyramid_images.append(gaussian_images[level-1])
        else:
            h = gaussian_images[i].shape[0]
            w = gaussian_images[i].shape[1]
            la = gaussian_images[i] - resize(gaussian_images[i+1],h,w)
            pyramid_images.append(la)
    return pyramid_images
'''
def laplacian_pyramid(gaussian_images,level):
    pyramid_images = []
    la_spectrum = []
    for i in range(level):
        la = gaussian_images[i] - gaussian(gaussian_images[i])
        pyramid_images.append(la)
        sp = magnitude_spectrum(la)
        la_spectrum.append(sp)
    return pyramid_images, la_spectrum

level = 5
path = '../hw2_data/task1and2_hybrid_pyramid/'
file_name = '4_einstein'
img = cv2.imread(path+file_name+'.bmp',0)
#img = cv2.imread(file+'.bmp',0)

py, sp = gaussian_pyramid(img,level)
la, la_sp = laplacian_pyramid(py,level)



for i in range(level):

    plt.subplot(5,4,4*i+1)
    if i==0: 
        plt.title('Gaussian')
    plt.axis('off')
    plt.imshow(py[i], cmap = 'gray') 
    
    plt.subplot(5,4,4*i+2)
    if i==0:
        plt.title('Spectrum')
    plt.axis('off')
    plt.imshow(sp[i], cmap = 'gray') 
    
    plt.subplot(5,4,4*i+3)
    if i==0:
        plt.title('Laplacian')
    plt.axis('off')
    plt.imshow(la[i], cmap = 'gray') 
    
    plt.subplot(5,4,4*i+4)
    if i==0:
        plt.title('Spectrum')
    plt.axis('off'), plt.imshow(la_sp[i], cmap = 'gray') 
    
plt.tight_layout()
plt.savefig(file_name +  '.png')
plt.show() 
   

'''
file_name = 'hw2_data/task1and2_hybrid_pyramid/1_bicycle'
img = cv2.imread(file_name+'.bmp',0)
img = subsampling(img)

plt.imshow(img, cmap = 'gray')
plt.show()
'''

