# import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv


WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + './dice_groups/'

class Assignment:
    def __init__(self, img_file_path): 
       self.img_file_path = img_file_path
       self._bgr = self.init()
       self._rgb = self._bgr[:,:,::-1]
       self._gray = self.init_gray()
       
       
       self.videoCapture()
       preload = self.preload_img()
       self.show(preload)
       
    def videoCapture (self):
        # # 先抓攝影機
        # cap = cv.VideoCapture(0)
        # if(cap):
        #     print('OK')
        pass
       
    def gamma (self, img, gamma_value):
        return np.power(img/float(np.max(img)), gamma_value)
       
        
    def get_gray(self,img):
        return cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    
    def init(self):
         return cv.imread(self.img_file_path)
       
    def init_gray(self):
        return cv.imread(self.img_file_path, cv.IMREAD_GRAYSCALE)
        
    
    def canny(self,img):
        return cv.Canny(img,100,200)
    
    def show(self,img):
        plt.imshow(img)
        
    def template_pattern(self,string_number):
        pass
        
    
    def preload_img (self):
        gamma = self.gamma(self._gray,1.5)
        
        # 把原圖灰階後做成canny
        return self._gray
       

if __name__ == "__main__":
    a = Assignment(WORK_DIR + './diss.png')
    