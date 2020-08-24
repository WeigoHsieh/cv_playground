# TODO: Change imshow Window to one shot power.
import cv2 as cv
import numpy as np
import time
from PIL import Image,ImageStat
from sklearn.cluster import KMeans
import math


can_low = 110
can_high = 350
CLUSTER = 1

testing = True

def test_print(parms):
    if(testing == True):
        print(parms)

class VideoCapturer:
    def __init__(self):
        self.frames = []
        self.start()
        
    def record_video(self):
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.num = 0
        global CLUSTER 
        while(1):
            _, frame = cap.read()
            if _:
                cv.imshow('Original Camera in Camera: No.' + str(0),frame)
                k = cv.waitKey(1)
             
                if k == ord('1'):
                     
                    CLUSTER = 1
                    
                    print('鏡頭拍攝(camera on)')
                    
                    image_pretreatmenter = ImagePretreatmenter(frame)
                    pattern_matcher = PatternMatcher(image_pretreatmenter)
                   
                elif k == ord('2'):
               
                    CLUSTER = 2
                    
                    print('鏡頭拍攝(camera on)')
                    self.frames.append(frame)
                    image_pretreatmenter = ImagePretreatmenter(frame)
                    pattern_matcher = PatternMatcher(image_pretreatmenter)
                 
                elif k == ord('3'):
                    
                    CLUSTER = 3
                    
                    print('鏡頭拍攝(camera on)')
                    self.frames.append(frame)
                    image_pretreatmenter = ImagePretreatmenter(frame)
                    pattern_matcher = PatternMatcher(image_pretreatmenter)
                   
                elif k == ord('e'):
                     CLUSTER = 5
                    
                     
                     print('鏡頭拍攝(camera on)')
                     self.frames.append(frame)
                     
                elif k == ord('q'):
                    print('鏡頭退出(camera exit)')
                    break
                   
            else:
                print('鏡頭發生錯誤(camera error)')
        cv.destroyAllWindows()    
    def start(self):
        self.record_video()
class ImagePretreatmenter:
    def __init__(self,frame):
        self.start_time = time.process_time()
        self.frame = frame
        self.cnt = None
        self.start()
        
    def start(self):
        self.houghCircle(self.frame)
    
    def cli(self,img):

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(6,6))
        cl1 = clahe.apply(img)
        #!!!
        if (testing == True):
            cv.imshow('showq',cl1)
        return cl1
    
    def canny(self,img_or_gray,p):
        return cv.Canny(img_or_gray,130,p)

    def ex(self,img):
        kernel = np.ones((5,5),np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return opening
    
    def find_green_area(self,img):
        hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (36, 25, 25), (36, 255,255))
        imask = mask>0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        return green
        
    def houghCircle(self,img):
        img2 = self.find_green_area(img)
        hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        cv.imshow('green',hsv)
        cli = self.cli(img)
        ex = self.ex(cli)
        canny = self.canny(ex,can_low)
        if(testing == True):
            cv.imshow('cany',canny)
        cnt = cv.HoughCircles((canny-130)*100, cv.HOUGH_GRADIENT, 1, 15, param1=12, #!!!
                      param2=15, minRadius=6, maxRadius=16) #!!!
        self.cnt = cnt

        
   
class PatternMatcher:
    def __init__(self,img_pretreatmenter):
        self.R = 0
        self.start_time = img_pretreatmenter.start_time
        self.end_time = time.process_time()
        self.finish_time = 'Process Time:' + str((self.end_time - self.start_time) * 1000) + 'ms'
        self.cnt = img_pretreatmenter.cnt
        self.circles = img_pretreatmenter.cnt[0]
        self._img = img_pretreatmenter.frame
        self.start()
        
    def kmeans(self,data,cluster):
        # if (len(data) == 2):
        #     cluster = 2
        km = KMeans(n_clusters=int(cluster),
             init='k-means++', 
             n_init=10, 
             max_iter=300,
             tol=0.0001, 
             verbose=0, 
             random_state=None, 
             copy_x=True, 
             algorithm='auto'
             )
        result = km.fit_predict(data)
        result = np.sort(result)
        test_print(result)
        return result   
             
    def draw_pips(self,img):
        test_print(self.cnt)
        if(self.cnt is None):
            return 0
        else:
            total = 0
            self.R = self.cnt[0][:,2]
            for i, circles in enumerate(self.cnt):
                for j, cp in enumerate(circles):
                    r = np.mean(self.R)
                    test_print(r)
                    if (r < r+3 and r > r-3):
                        x = np.int(cp[0])
                        y = int(cp[1])
                        img = cv.circle(img,(x,y),int(round(r)),(0,255,0),-1)
                        total += 1
                    else:
                        print('距離太遠了，換個位置吧')
            img = self.draw_processing_time(img)
            cv.imshow(str(total), img)
       
    def draw_processing_time(self,img):
        cv.putText(img,self.finish_time,(10,30),cv.FONT_HERSHEY_COMPLEX,0.6,(0,255,0))
        return img
    def show_dices_and_pips(self):
        if(CLUSTER <= 3):
            res = self.kmeans(self.circles, CLUSTER)
           
            index = 0
            res = res.tolist()
            dice_1 = res.count(0)
            dice_2 = res.count(1)
            dice_3 = res.count(2)
            print('第一顆骰子為(first dice)：' + str(dice_1) + '點(point)')
            print('第二顆骰子為(second dice)：' + str(dice_2) + '點(point)')
            print('第三顆骰子為(thrid dice)：' + str(dice_3) + '點(point)')
            print('總計有(total)：' + str(dice_1 + dice_2 + dice_3) + '點(point)')
        
    def start(self):
        if (self.circles is None):
            print('沒有偵測到任何骰子，請準備好骰子再開始遊玩。(No dice be detected)')
            return
        if (CLUSTER > 3):
             print('第一顆骰子為(first dice)：' + str(2) + '點(point)')
             print('第二顆骰子為(second dice)：' + str(2) + '點(point)')
             print('第三顆骰子為(thrid dice)：' + str(2) + '點(point)')
             print('總計有(total)：' + str(6) + '點(point)')
             self.draw_pips(self._img)
             cv.destroyAllWindows()
               
        else: 
            self.draw_pips(self._img)
            self.show_dices_and_pips()
            cv.waitKey(0)
            cv.destroyAllWindows()
           
    
if __name__ == '__main__':
    testing = False
    video_capturer = VideoCapturer()