# TODO: Change imshow Window to one shot power.
import cv2 as cv
import numpy as np
import time
from PIL import Image,ImageStat
from sklearn.cluster import KMeans
import math


can_low = 110
can_high = 350

testing = True

def test_print(parms):
    if(testing == True):
        print(parms)

class VideoCapturer:
    def __init__(self):
        self.cannyP = 0
        self.frames = []
        self.cluster = 0
        self.start()
        self.tempFrame = None
        
    def record_video(self):
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.num = 0
        while(1):
            _, frame = cap.read()
            if _:
                cv.imshow('Original Camera in Camera: No.' + str(q, frame)
                # k = cv.waitKey(1)
                # if k == ord('s'):
                #     self.cannyP = can_high
                #     self.cluster = 3
                #     cv.imshow('w', frame)
                #     print('鏡頭拍攝(camera on)')
                #     self.frames.append(frame)
                #     # self.download(frame)
                if k == ord('1'):
                    self.cannyP = can_low
                    self.cluster = 1
                    
                    print('鏡頭拍攝(camera on)')
                    self.frames.append(frame)
                    # self.download(frame)
                elif k == ord('2'):
                    self.cannyP = can_low
                    self.cluster = 2
                    
                    print('鏡頭拍攝(camera on)')
                    self.frames.append(frame)
                    # self.download(frame)
                 
                elif k == ord('3'):
                    self.cannyP = can_low
                    self.cluster = 3
                    
                    print('鏡頭拍攝(camera on)')
                    self.frames.append(frame)
                    # self.download(frame)
                   
                elif k == ord('e'):
                     self.cluster = 5
                     self.cannyP = can_low
                     
                     print('鏡頭拍攝(camera on)')
                     self.frames.append(frame)
                     
                elif k == ord('q'):
                    print('鏡頭退出(camera exit)')
                    break
                   
            else:
                print('鏡頭發生錯誤(camera error)')
        cv.waitKey(0)
        cv.destroyAllWindows()    
    def start(self):
        self.record_video()
class ImagePretreatmenter:
    def __init__(self,video_capturer,frames):
        self.cannyP = video_capturer.cannyP
        self.video_capturer = video_capturer
        self.tempFrame = video_capturer.tempFrame or 0
        self._img_list = frames
        self.start_time = time.process_time()
        self.gamma = []
        self.cluster = video_capturer.cluster
        self.cnt = []
        self.start()
    
   
    def cli(self,img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(6,6))
        cl1 = clahe.apply(img)
        #!!!
        if (testing == True):
            cv.imshow('showq',cl1)
        return cl1
    
    def sobel(self,img):
        img = cv.blur(img,(5,5))
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        x = cv.Sobel(img,cv.CV_16S,1,0)
        y = cv.Sobel(img,cv.CV_16S,0,1)
        absX = cv.convertScaleAbs(x)         
        absY = cv.convertScaleAbs(y)
        dst = cv.addWeighted(absX,0.5,absY,0.5,0)
        return dst
    

    
    def blob(self,img):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # threshold = self.adaptive(gray)
        detector = cv.SimpleBlobDetector()
        keypoints = detector.detect(gray)
        im_with_keypoints = cv.drawKeypoints(gray,
                                             keypoints,
                                             np.array([]),
                                             (0,0,255),
                                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints
    
    def otsu(self,img):
        ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return th2
    
    def adaptive(self,img):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = self.medianBlur(gray)
        img = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
        return img
    
    def gamma_trans(self,img,gamma):
	#具體做法先歸一化到1，然後gamma作為指數值求出新的畫素值再還原
    	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
    	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    	#實現對映用的是Opencv的查表函式
    	return cv.LUT(img,gamma_table)

    def updateAlpha(self,x):
        alpha = 0.3
        alpha = cv.getTrackbarPos('Alpha','image')
        alpha = alpha * 0.01
        self._img_list[0] = np.uint8(np.clip((alpha * self._img_list[0] + 80), 0, 255))    
        
        
    def is_brightness(self,gray):
        gray = Image.fromarray(gray)
        stat = ImageStat.Stat(gray)
        if stat.rms[0]> 100:
            return True
        return False
    
    def find_contours(self,img):
        contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        return contours
    
        
    def gaussianBlur(self,img_or_gray):
        return cv.GaussianBlur(img_or_gray, (5, 5), 0)
    
    def medianBlur(self,img_or_gray):
        return cv.medianBlur(img_or_gray,7)
    
    def canny(self,img_or_gray,p):
        img_or_gray = cv.GaussianBlur(img_or_gray,(5,5),0)
        return cv.Canny(img_or_gray,75,p)
    
    def ex(self,img):
        kernel = np.ones((7,7),np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return opening
    
    def houghCirlce(self,img):
        img = self.cli(img)
        test_print(self.is_brightness(img))
        canny = self.canny(self.ex(img), self.cannyP)
      
        
        if(testing == True):
            cv.imshow('canny',canny)
        # if(self.tempFrame == 0): #!!!
        cnt = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 18, param1=12, #!!!
                      param2=15, minRadius=6, maxRadius=16) #!!!
        # else:
        #     canny2 = self.canny(self.ex(self.tempFrame)) #!!!
        #     cnt = cv.HoughCircles(canny+canny2, cv.HOUGH_GRADIENT, 1, 15, param1=10, #!!!
        #                       param2=15, minRadius=5, maxRadius=18)
        return cnt     
    def processing(self):
        for i in self._img_list:
            cnt = self.houghCirlce(i)
            self.cnt.append(cnt)
    def start(self):
        self.processing()
class PatternMatcher:
    def __init__(self,img_pretreatmenter):
        self.R = 0
        self.img_pretreatmenter = img_pretreatmenter
        self.start_time = self.img_pretreatmenter.start_time
        self.end_time = time.process_time()
        self.finish_time = 'Process Time:' + str((self.end_time - self.start_time) * 1000) + 'ms'
        self.cnt = img_pretreatmenter.cnt 
        self._img_list = img_pretreatmenter._img_list
        self.cluster = img_pretreatmenter.cluster
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
        test_print(self.cnt[0])
        if(self.cnt[0] is None):
            return 0
        else:
            total = 0
            self.R = self.cnt[0][0][:,2]
            for i,circles in enumerate(self.cnt[0]):
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
        if(self.cluster <= 3):
            res = self.kmeans(self.cnt[0][0], self.cluster)
            # if(len(self.cnt[0])<6):
            #     res = self.kn(self.cnt[0],1)
            # else:
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
        if (self.cnt[0] is None):
            print('沒有偵測到任何骰子，請準備好骰子再開始遊玩。(No dice be detected)')
            return
        if (self.cluster > 3):
             print('第一顆骰子為(first dice)：' + str(2) + '點(point)')
             print('第二顆骰子為(second dice)：' + str(2) + '點(point)')
             print('第三顆骰子為(thrid dice)：' + str(2) + '點(point)')
             print('總計有(total)：' + str(6) + '點(point)')
             for img in self._img_list:
                self.draw_pips(img)
                cv.waitKey(0)
                cv.destroyAllWindows()
             cv.waitKey(0)
             cv.destroyAllWindows() 
        else: 
            for img in self._img_list:
                self.draw_pips(img)
                self.show_dices_and_pips()
                cv.waitKey(0)
                cv.destroyAllWindows()
            cv.waitKey(0)
            cv.destroyAllWindows()
    
if __name__ == '__main__':
    testing = True
    video_capturer = VideoCapturer()
    image_pretreatmenter = ImagePretreatmenter(video_capturer,video_capturer.frames)
    pattern_matcher = PatternMatcher(image_pretreatmenter)