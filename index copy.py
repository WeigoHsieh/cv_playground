# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import cv2 as cv
# import tensorflow.examples.tutorials.mnist
import time
import math
from sklearn.cluster import KMeans

WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + 'dice_groups/'
DICE_WRITE_DIR = WORK_DIR + 'diceing/5/'


class VideoCapturer:
    def __init__(self, camera):
        self.frame = []
        self.start(camera)

    def start(self, camera):
        cap = cv.VideoCapture(camera, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.num = 0
        while(1):
            _, frame = cap.read()
            cv.imshow('Original Camera in Camera: No.' + str(camera), frame)
            k = cv.waitKey(1)
            if k == ord('s'):
                cv.imshow('w', frame)
                self.frame.append(frame)
                # self.download(frame)
            elif k == ord('q'):
                break
        cv.destroyAllWindows()
        cap.release()

    def download(self, frame):
        cv.imwrite(DICE_WRITE_DIR + str(self.num)+'.png', frame)
        print(self.num)
        self.num += 1

class ImagePretreatmenter:
    def __init__(self, img_list):
        self.after_pretreatment_list = []
        self._img_list = img_list
        self.ares = []
        start = time.process_time()
        self.start()
        end = time.process_time()
        print('花費了：' + str((end - start)*100) + '毫秒')

    def start(self):
        num = 0
        for i in self.processing():
            cv.namedWindow('Photo:'+str(num))
            # testing
            cv.imshow('Photo:'+str(num), i)
            self.after_pretreatment_list.append(i)
            num += 1
            cv.waitKey()
            cv.destroyAllWindows()

    def medianBlur(self, img):
        return cv.medianBlur(img, 9)
    
    def drawHoughLine(self,img):
        pass

    def adaptiveGaussianThresHolding(self, img):
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
       
    def kn(self,data):
        km = KMeans(n_clusters=3,
             init='k-means++', 
             n_init=10, 
             max_iter=150, 
             tol=0.0001, 
             verbose=0, 
             random_state=None, 
             copy_x=True, 
             n_jobs=1, 
             algorithm='auto'
             )
        print(km)
        result = km.fit_predict(data)
        return result
     # 灰階
    def gray(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def ex(self,img):
        kernel = np.ones((5,5),np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return opening
    
    # 邊緣檢測
    def canny(self, img):
        return cv.Canny(img, 120, 350) 
    # 120,350 200,400

    # 高斯模糊
    def GaussianBlur(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)


    def adjust(self,cnt,arr):
        for c in cnt:
            print(c)
            r = c[2]
            print(r)
            if (r < 15 and r > 5):
                yield [c[0],c[1],c[2]]

    # 霍夫曼圓形檢測(灰值)
    def houghCircle(self, img):
        gray = self.gray(img)
        gau = self.GaussianBlur(gray)
        med = self.medianBlur(gray)
        canny = self.canny(self.ex(img))
        
        cnt = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 11, param1=10,
                          param2=15, minRadius=5, maxRadius=30)  # 把半徑範圍縮小點，檢測內圓，瞳孔
        total = 0
        if(cnt is None):
            print('找不到')
            return
        else:
            for circles in cnt:
                for cp in circles:
                    r = int(cp[2])
                    x = np.int(cp[0])
                    y = int(cp[1])
                    # print(self.pip_distance(x,y))
                    img = cv.circle(img,(x,y),11,(0,255,0),-1)
                    total += 1
                    
        res = self.kn(cnt[0])
        res = res.tolist()
        dice_1 = res.count(0)
        dice_2 = res.count(1)
        dice_3 = res.count(2)
        print('第一顆骰子為：' + str(dice_1) + '點')
        print('第二顆骰子為：' + str(dice_2) + '點')
        print('第三顆骰子為：' + str(dice_3) + '點')
        print('總計有：' + str(len(res)) + '點')
        return img
   # 輪廓檢測(需要灰、模糊、或者二值化)
    def contours(self, img):
        clone = img.copy()
        gray = self.gray(img)
        gau = self.GaussianBlur(gray)
        hou = self.houghCircle(img)
        km = self.kn([[1,2],[1,2],[3,4]])
        return hou

    def processing(self):
        for i in self._img_list:
            cont = self.contours(i)
            yield cont

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
            TEMPLATE_GROUP_DIR = TEMPLATE_GROUP + 'dice_' + str(i+1) + '.png'
            print(TEMPLATE_GROUP_DIR, '\n')
            img = cv.imread(TEMPLATE_GROUP)
            # cv.imshow('',img)
        # destory()

    preload_test()
  
    
  
    cv.waitKey(0)
    cv.destroyAllWindows()
 