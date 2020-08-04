# import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv
# import tensorflow.examples.tutorials.mnist

WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + './dice_groups/'

class Assignment:
    def __init__(self, img_file_path): 
       # self.img_file_path = img_file_path
       # self._bgr = self.init()
       # self._rgb = self._bgr[:,:,::-1]
       # self._gray = self.get_gray(self._bgr)
       
       
       # self.videoCapture()
       # preload = self.preload_img()
       # # self.show_gray(preload)
       
       # self.template_pattern(self._gray,'6')
       self.videoCapture()
       
       
    def videoCapture (self):
        cap = cv.VideoCapture(0)
        cap.set(0,480)
        while(1):
            frame = cap.read()
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            lower_red = np.array([30,150,50]) 
            upper_red = np.array([255,255,180]) 
            mask = cv.inRange(hsv, lower_red, upper_red) 
            res = cv.bitwise_and(frame,frame, mask= mask)
            kernel = np.ones((15,15),np.float32)/225 
            smoothed = cv.filter2D(res,-1,kernel) 
            cv.imshow('Original',frame) 
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows() 
        cap.release()
      

       
    def gamma (self, img, gamma_value):
        return np.power(img/float(np.max(img)), gamma_value)
       
        
    def get_gray(self,img):
        return cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    def init(self):
         return cv.imread(self.img_file_path)
       
    def init_gray(self):
        return cv.imread(self.img_file_path)
        
    
    def canny(self,img):
        return cv.Canny(img,100,200)
    
    def show(self,img):
        plt.imshow(img)
        
    def adaptiveThreshold(self,img):
        return cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
            
            
    def matchTemplate(self,img,templ):
       pass
        
        # TEMPLATE_GROUP+'./dice_' + string_number+ '.png'
    def template_pattern(self,img,string_number):
        TEMPLATE = cv.imread('ref.png',0)
        plt.imshow(TEMPLATE)
        print(TEMPLATE.shape)
        # self.matchTemplate(img,TEMPLATE)
        # 紀錄模板尺寸
        
        w, h = TEMPLATE.shape[::-1]
        print(w,h)
        # 做模型比對(出來的值為ndarray float32)
        res = cv.matchTemplate(img, TEMPLATE, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold) 
        for pt in zip(*loc[::-1]):  # *號表示可選參數
            cv.rectangle(self._rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
        cv.imshow('Detected',self._rgb)
 
        
        
    def otsu (self,img):
        blur = self.gaussianBlur(img)
        return cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    def gaussianBlur (self,img):
        return cv.GaussianBlur(img,(5,5),0)
        
    def show_gray(self,img):
        plt.imshow(img,cmap='gray')
    
    def preload_img (self):
        #高斯模糊降噪
        gau = self.gaussianBlur(self._gray)
        adaptive = self.adaptiveThreshold(gau)
        # otsu = self.otsu(self._gray)
        # 把原圖灰階後做成canny
        return adaptive
       

if __name__ == "__main__":
    a = Assignment(WORK_DIR + './diss2.png')
    