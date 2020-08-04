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
       self._gray = self.get_gray(self._bgr)
       
       
       self.videoCapture()
       preload = self.preload_img()
       # self.show_gray(preload)
       
       self.template_pattern(preload,'1')
       
       
    def videoCapture (self):
        # # 先抓攝影機
        # cap = cv.VideoCapture(0)
        # if(cap):
        #     print('OK')
        pass
       
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
            
            
    def matchTemplate(img,templ):
        # 紀錄模板尺寸
        h, w = templ.shape[:2]
        res = cv.matchTemplate(img, templ, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)  # 匹配程度大於%80的坐標y,x
        for pt in zip(*loc[::-1]):  # *號表示可選參數
            right_bottom = (pt[0] + w, pt[1] + h)
            cv.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
        
        
    def template_pattern(self,img,string_number):
        template = cv.imread(TEMPLATE_GROUP+'./dice_' + string_number+ '.png',0)
        matchTemplate(img,template)
        
        
        # 會得到座標(四個角)
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.9
        
        img2 = img.copy()
        
        
        
        #推算座標
        
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img,top_left, bottom_right, 255, 2)
        
        
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        
        
        # plt.imshow(img)
        
        # for pt in zip(*loc[::-1]): 
        #     cv.rectangle(self._rgb, pt, (pt[0] + w, pt[1] + h), (7,249,151), 2)
        #     #顯示圖像
        #     plt.imshow(self._rgb)
        #     # cv.waitKey(0)
        #     # cv.destroyAllWindows()
     


        
        
    def otsu (self,img):
        blur = self.gaussianBlur(img)
        return cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    def gaussianBlur (self,img):
        return cv.GaussianBlur(img,(5,5),0)
        
    def show_gray(self,img):
        plt.imshow(img,cmap='gray')
    
    def preload_img (self):
        #高斯模糊降噪音
        gau = self.gaussianBlur(self._gray)
        adaptive = self.adaptiveThreshold(gau)
        # otsu = self.otsu(self._gray)
        # 把原圖灰階後做成canny
        return adaptive
       

if __name__ == "__main__":
    a = Assignment(WORK_DIR + './diss2.png')
    