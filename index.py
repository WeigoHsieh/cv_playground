# import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv


WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + './dice_groups/'

class Assignment:
    def __init__(self, img_file_path): 
       self.img_file_path = img_file_path
       self._rgb = self.init()
       self._gray = self.get_gray(self._rgb)
       
       
       self.videoCapture()
       preload = self.preload_img()
       self.show(preload)
       
    def videoCapture (self):
        pass
       
    def get_gray(self,img):
        return cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    def init(self):
        return cv.imread(self.img_file_path)
    def canny(self,img):
        return cv.Canny(img,100,200)
    
    def show(self,img):
        plt.imshow(img)
    
    def preload_img (self):
        # 把原圖灰階後做成canny
        return self.canny(self._gray)
        
       

if __name__ == "__main__":
    a = Assignment(WORK_DIR + './diss.png')
    