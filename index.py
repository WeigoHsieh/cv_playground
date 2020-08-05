# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# import tensorflow.examples.tutorials.mnist

WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + './dice_groups/'
DICE_WRITE_DIR = WORK_DIR + './diceing/5/'


class VideoCapturer:
    def __init__(self, camera):
        self.frame = []
        self.start(camera)

        print(self.frame)

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
                print(len(self.frame))
                print(self.num)
        
            elif k == ord('q'):
                break
        cv.destroyAllWindows()
        cap.release()
        print(self.frame)
        
    def download(self,frame):
        cv.imwrite(DICE_WRITE_DIR + str(self.num)+'.png', frame)
        print(self.num)
        self.num += 1
        
    


class ImagePretreatmenter:
    def __init__(self, img_list):
        self.after_pretreatment_list = []
        print('---進入圖像前處理---')
        self._img_list = img_list
        print(self._img_list)
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
        return cv.Canny(img,1,5)
    
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
        res = match(self.method)
        return res
    def match(self,method):
        return cv.matchTemplate(self.gray, self.template, method)
    

class PatternMatcherTest:
    def __init__(self,template, origin_img):
        self.template = cv.imread(template)
        self.origin_img = cv.imread(origin_img)
        self.gray = cv.cvtColor(self.origin_img, cv.COLOR_RGB2GRAY)
        self.start()
        
        
    def start(self):
        
        res = self.detect(self.origin_img, self.template)
        print( type(res) )
    
    def detect(self,origin,templ):
        # //matchTemplate會回傳座標
        return cv.matchTemplate(self.gray, self.template, cv.TM_CCOEFF_NORMED)
    
    
class DiceImageTrainer: 
    def __init__(self,pretreatmenter):
        self._pretreatmenter = pretreatmenter
        
    def setup():
        pass

    

class Assignment:
    def __init__(self, img_file_path):
        self.img_file_path = img_file_path
        self.frame = []
        self._bgr = self.init()
        self._rgb = self._bgr[:,:,::-1]
        self._gray = self.get_gray(self._bgr)

        self.videoCapture()
        preload = self.preload_img()
        self.show_gray(preload)

        self.template_pattern(self._gray,'6')
        self.videoCapture()
        print(self.frame)

    def videoCapture(self):
        cap = cv.VideoCapture(0)
        while(1):
            _, frame = cap.read()
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            lower_red = np.array([30, 150, 50])
            upper_red = np.array([255, 255, 180])
            mask = cv.inRange(hsv, lower_red, upper_red)
            res = cv.bitwise_and(frame, frame, mask=mask)
            kernel = np.ones((15, 15), np.float32)/225
            smoothed = cv.filter2D(res, -1, kernel)
            cv.imshow('Original', frame)
            k = cv.waitKey(1)
            if k == ord('s'):
                cv.imshow('w', frame)
                self.frame.append(frame)
            elif k == ord('q'):
                break
        cv.destroyAllWindows()
        cap.release()

    def gamma(self, img, gamma_value):
        return np.power(img/float(np.max(img)), gamma_value)

    def get_gray(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def init(self):
        return cv.imread(self.img_file_path)

    def init_gray(self):
        return cv.imread(self.img_file_path)

    def canny(self, img):
        return cv.Canny(img, 100, 200)

    def show(self, img):
        plt.imshow(img)

    def adaptiveThreshold(self, img):
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, 11, 2)

    def matchTemplate(self, img, templ):
        pass

        # TEMPLATE_GROUP+'./dice_' + string_number+ '.png'
    def template_pattern(self, img, string_number):
        TEMPLATE = cv.imread('ref.png', 0)
        plt.imshow(TEMPLATE)
        print(TEMPLATE.shape)
        # self.matchTemplate(img,TEMPLATE)
        # 紀錄模板尺寸

        w, h = TEMPLATE.shape[::-1]
        print(w, h)
        # 做模型比對(出來的值為ndarray float32)
        res = cv.matchTemplate(img, TEMPLATE, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):  # *號表示可選參數
            cv.rectangle(
                self._rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        cv.imshow('Detected', self._rgb)

    def otsu(self, img):
        blur = self.gaussianBlur(img)
        return cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    def gaussianBlur(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)

    def show_gray(self, img):
        plt.imshow(img, cmap='gray')

    def preload_img(self):
        # 高斯模糊降噪
        gau = self.gaussianBlur(self._gray)
        adaptive = self.adaptiveThreshold(gau)
        # otsu = self.otsu(self._gray)
        # 把原圖灰階後做成canny
        return adaptive


if __name__ == "__main__":
    video = VideoCapturer(0)
    imagePretreatmenter = ImagePretreatmenter(video.frame)
    # patternMatcher = PatternMatcher(TEMPLATE_GROUP + './dice_5.png', WORK_DIR + './diss2.png')
    print()