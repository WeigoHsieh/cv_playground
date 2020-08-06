# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import cv2 as cv
# import tensorflow.examples.tutorials.mnist

WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + 'dice_groups/'
DICE_WRITE_DIR = WORK_DIR + 'diceing/5/'


class VideoCapturer:
    def __init__(self, camera):
        self.frame = []
        self.start(camera)


    def start(self, camera):
        cap = cv.VideoCapture(camera,cv.CAP_DSHOW)
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
             cv.waitKey()
             cv.destroyAllWindows()
        
        
    #自適應高斯模糊濾波器    
    def adaptiveGaussianThresHolding(self,img):
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, 11, 2)
    #灰階
    def gray(self,img):
        return cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    def sobel (self,img):
        x = cv.Sobel(img, cv.CV_16S, 1, 0)
        y = cv.Sobel(img, cv.CV_16S, 0, 1)
        
        absX = cv.convertScaleAbs(x)# 轉回uint8
        absY = cv.convertScaleAbs(y)
        
        dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        return dst
    
    #邊緣檢測
    def canny(self,img):
        return cv.Canny(img,200,400)
    
    #高斯模糊
    def GaussianBlur(self,img):
        return cv.GaussianBlur(img,(9,9),0)
    
    # 拉普拉斯算子
    def lapl(self,img):
        return cv.Laplacian(img,cv.CV_64F)
    
    # 銳利
    def curve(self,img):
         kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]], np.float32)
         dst = cv.filter2D(img, -1, kernel=kernel)
         return dst
     
    # 霍夫曼圓形檢測(灰值)
    def houghCircle(self,img):
        gray = self.gray(img)
        gau = self.GaussianBlur(gray)
        circles = cv.HoughCircles(gau,cv.HOUGH_GRADIENT,1,12,
                                  param1=50,param2=30,minRadius=20,maxRadius=40)
        return circles
     
        
     #輪廓檢測(需要灰、模糊、或者二值化)
    def contours (self,img):
        gray = self.gray(img)
        gau = self.GaussianBlur(gray)
        # binaryImg = self.sobel(gau)
        # cv.imshow('',binaryImg)
        canny = self.canny(gau)
        hou = self.houghCircle(img)
        # sobel = self.sobel(gray)
        # method = cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        cnts, _ = cv.findContours(canny.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        clone = img.copy()
        cv.drawContours(clone,cnts,-1,(0,255,0),2)
        count = 0
        ares_avrg = 0
        for cont in cnts:
            ares = cv.contourArea(cont)#計算包圍性狀的面積
            if ares<50:   #過濾面積小於10的形狀
                continue
            count+=1    #總體計數加1
            ares_avrg+=ares
        
        print('總共有：' + str(count) +'點')
        
        return clone

    
    def processing(self):
       # cv.waitKey(0)
       # cv.destroyAllWindows()
       for i in self._img_list:
         gray = self.gray(i)    
         sobel = self.sobel(gray)
         contours = self.contours(i)
        
         yield contours
    
    #圖片裁減   
    def cut(self, img,x,y,w,h):
        crop_img = img[y:y+h,x:x+w]
        return crop_img
    
    def blob_dection(self,img):
        detector = cv.SimpleBlobDetector()
        keypoints = detector.detect(img)
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img_with_keypoints
    

# # Detect blobs.


# # Draw detected blobs as red circles.

# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)
    

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
        return cv.matchTemplate(self.gray, self.templ, method)
    def detect(self):
        pass
    
    def test(self):
        self.templ = cv.imread(WORK_DIR + 'blob.png')
        #化緣
        mark(self.origin,self.templ)
        
    def mark(self,origin_gray,templ):
        # //matchTemplate會回傳座標
        # return
        w,h = templ.shape[::-1]
        res = cv.matchTemplate(origin_gray, templ, cv.TM_CCOEFF_NORMED)
        loc = np.where (res >= self.threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(self.origin_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
      
    

class PatternMatcherTest:
    def __init__(self,template, origin_img):
        self.template = cv.imread(template)
        self.origin_img = cv.imread(origin_img) 
        self.gray = cv.cvtColor(self.origin_img, cv.COLOR_RGB2GRAY)
        self.threshold = 0.8
        self.start()
    def blob_dection(self,img):
        pass
#         detector = cv2.SimpleBlobDetector()
#         keypoints = detector.detect(img)

    

# # Detect blobs.


# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)




        
        
    def start(self):
        self.detect(self.gray,self.template)
        print(self.template,self.gray)
        # cv.imshow('',self.origin_img)
        
    def detect(self,origin_gray,templ):
        pass
        # //matchTemplate會回傳座標
        # return
        # w,h = templ.shape[::-1]
        # res = cv.matchTemplate(origin_gray, templ, cv.TM_CCOEFF_NORMED)
        # loc = np.where (res >= self.threshold)
        # for pt in zip(*loc[::-1]):
        #     cv.rectangle(self.origin_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
      
        
    
class DiceImageTrainer: 
    def __init__(self,pretreatmenter):
        self._pretreatmenter = pretreatmenter
    

if __name__ == "__main__":
    
    def destory():
        cv.waitkey()
        cv.destroyAllWindows()
    
    def patternmatch_test():
        pass
    
    def preload_test():
        video = VideoCapturer(0)
        imagePretreatmenter = ImagePretreatmenter(video.frame)
        
    def template_preload_saobel():
        for i in range(6):
            TEMPLATE_GROUP_DIR = TEMPLATE_GROUP +'dice_'+ str(i+1) + '.png'
            print(TEMPLATE_GROUP_DIR,'\n')
            img = cv.imread(TEMPLATE_GROUP)
            # cv.imshow('',img)
        # destory()
                
    
            
            
    preload_test()
    # template_preload_saobel()
    
    # video = VideoCapturer(0)
    # imagePretreatmenter = ImagePretreatmenter(video.frame)
    # patternMatcher = PatternMatcherTest(WORK_DIR + './blob.png', WORK_DIR + 'origin.png')
    cv.waitKey(0)
    cv.destroyAllWindows()
    # a = cv.imread(TEMPLATE_GROUP + './dice_ref.png')
    # cv.imshow('',a)
    # cv2.rectangle(img, 左上(20, 60), 右下(120, 160), 顏色(0, 255, 0), 粗細2)
