# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import cv2 as cv
# import tensorflow.examples.tutorials.mnist

WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + './dice_groups/'
DICE_WRITE_DIR = WORK_DIR + './diceing/5/'


class VideoCapturer:
    def __init__(self, camera):
        self.frame = []
        self.start(camera)


    def start(self, camera):
        cap = cv.VideoCapture(camera)
        self.num = 0
        while(1):
            _, frame = cap.read()
            
            cv.imshow('Original Camera in Camera: No.' + str(camera) , frame)
            k = cv.waitKey(1)
            if k == ord('s'):
                cv.imshow('w', frame)
                
                self.frame.append(frame)
                # self.download(frame)
             
            elif k == ord('q'):
                break
        cv.destroyAllWindows()
        cap.release()
      
        
    def download(self,frame):
        cv.imwrite(DICE_WRITE_DIR + str(self.num)+'.png', frame)
        print(self.num)
        self.num += 1
        
    


class ImagePretreatmenter:
    def __init__(self, img_list):
        self.after_pretreatment_list = []
        self._img_list = img_list
        self.start()

    def start(self):
        num = 0
        for i in self.processing():
             cv.namedWindow('Photo:'+str(num))
             # testing
             cv.imshow('Photo:'+str(num),i)
             self.after_pretreatment_list.append(i)
             num +=1
        
    #自適應高斯模糊濾波器    
    def adaptiveGaussianThresHolding(self,img):
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 11, 2)
    #灰階
    def gray(self,img):
        return cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    #邊緣檢測
    def canny(self,img):
        return cv.Canny(img,85,125)
    
    #高斯模糊
    def GaussianBlur(self,img):
        return cv.GaussianBlur(img,(3,3),0)
    
    # 拉普拉斯算子
    def lapl(self,img):
        return cv.Laplacian(img,cv.CV_64F)
    
    # 銳利
    def curve(self,img):
         kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
         dst = cv.filter2D(img, -1, kernel=kernel)
         return dst
    
    def processing(self):
       for i in self._img_list:
         gray = self.gray(i)
         
        
         yield self.canny(gray)
        
    #圖片裁減   
    def cut(self, img,x,y,w,h):
        crop_img = img[y:y+h,x:x+w]
        return crop_img
    

# PatternMatcher
# @params 
#  origin: List
#  templ: List
# 

class PatternMatcher:
    def __init__(self,templ,origin):
        self.templ = templ
        self.origin = origin
        self.method = cv.TM_CCOEFF_NORMED
        
    def templateMatching(self):
        res = self.match(self.method)
        return res
    def match(self,method):
        return cv.matchTemplate(self.gray, self.template, method)
    def detect(self):
        pass
    

class PatternMatcherTest:
    def __init__(self,template, origin_img):
        self.template = cv.imread(template,0)
        self.origin_img = cv.imread(origin_img)
        self.gray = cv.cvtColor(self.origin_img, cv.COLOR_RGB2GRAY)
        self.threshold = 0.8
        self.start()
        
        
    def start(self):
        self.detect(self.gray,self.template)
        cv.imshow('',self.origin_img)
        
    def detect(self,origin_gray,templ):
        # //matchTemplate會回傳座標
        # return
        w,h = self.template.shape[::-1]
        res = cv.matchTemplate(self.gray, self.template, cv.TM_CCOEFF_NORMED)
        loc = np.where (res >= self.threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(self.origin_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
      
        
    
class DiceImageTrainer: 
    def __init__(self,pretreatmenter):
        self._pretreatmenter = pretreatmenter
    

if __name__ == "__main__":
    video = VideoCapturer(0)
    imagePretreatmenter = ImagePretreatmenter(video.frame)
    # patternMatcher = PatternMatcherTest(TEMPLATE_GROUP + './dice_ref.png', WORK_DIR + './diss2.png')
    # a = cv.imread(TEMPLATE_GROUP + './dice_ref.png')
    # cv.imshow('',a)
    # cv2.rectangle(img, 左上(20, 60), 右下(120, 160), 顏色(0, 255, 0), 粗細2)
