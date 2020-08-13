
import cv2 as cv
import numpy as np
import time
from PIL import Image
from PIL import ImageStat
from sklearn.cluster import KMeans

class VideoCapturer:
    def __init__(self,camera):
        self.camera = camera
        self.frames = []
        self.start()
        
    def start(self):
        cap = cv.VideoCapture(self.camera, cv.CAP_DSHOW)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.num = 0
        while(1):
            _, frame = cap.read()
            if _:
                cv.imshow('Original Camera in Camera: No.' + str(self.camera), frame)
                k = cv.waitKey(1)
                if k == ord('s'):
                    cv.imshow('w', frame)
                    print('鏡頭拍攝')
                    self.frames.append(frame)
                    # self.download(frame)

                elif k == ord('q'):
                    print('鏡頭退出')
                    break
            else:
                print('鏡頭發生錯誤')
                break
        cv.waitKey(0)
        cv.destroyAllWindows()    
class ImagePretreatmenter:
    def __init__(self,frames):
        self._img_list = frames
        self.start_time = time.process_time()
        self.gamma = []
        self.cnt = []
        self.start()
        
    def is_brightness(self,gray):
        gray = Image.fromarray(gray)
        stat = ImageStat.Stat(gray)
        if stat.rms[0]> 100:
            return True
        return False
    
    def set_gamma(self,params):
        self.gamma = params   
        
    def gaussianBlur(self,img_or_gray):
        return cv.GaussianBlur(img_or_gray, (5, 5), 0)
    
    def medianBlur(self,img_or_gray):
        return cv.medianBlur(img_or_gray,5)
    
    def canny(self,img_or_gray):
        img = self.medianBlur(img_or_gray)
        return cv.Canny(img,125,350)
    
    def ex(self,img):
        kernel = np.ones((5,5),np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return opening
    
    def houghCirlce(self,img):
        canny = self.canny(self.ex(img))
        cnt = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 10, param1=10,
                          param2=15, minRadius=5, maxRadius=30)  # 把半徑範圍縮小點，檢測內圓，瞳孔
        if cnt is None:
            cnt = np.zero(3,3)
            return cnt
        return cnt     
    def processing(self):
        for i in self._img_list:
            cnt = self.houghCirlce(i)
            self.cnt.append(cnt)
    def start(self):
        self.processing()
class PatternMatcher:
    def __init__(self,img_pretreatmenter):
        self.img_pretreatmenter = img_pretreatmenter
        self.start_time = self.img_pretreatmenter.start_time
        self.end_time = time.process_time()
        self.finish_time = 'Process Time:' + str((self.end_time - self.start_time) * 1000) + 'ms'
        self.cnt = img_pretreatmenter.cnt 
        self._img_list = img_pretreatmenter._img_list
        self.start()
    def kmeans(self,data,cluster):
        km = KMeans(n_clusters=cluster,
             init='k-means++', 
             n_init=10, 
             max_iter=300, 
             tol=0.0001, 
             verbose=0, 
             random_state=None, 
             copy_x=True, 
             n_jobs=1, 
             algorithm='auto'
             )
        result = km.fit_predict(data)
        return result                
    def draw_pips(self,img):
        total = 0
        for i,circles in enumerate(self.cnt[0]):
            for j, cp in enumerate(circles):
                r = int(cp[2])
                if (r < 13 and r >= 10):
                    x = np.int(cp[0])
                    y = int(cp[1])
                    img = cv.circle(img,(x,y),12,(0,255,0),-1)
                    # img.putText()
                    total += 1
                else:
                    self.cnt[0][i][j] = False
        img = self.draw_processing_time(img)
        cv.imshow(str(total), img)
       
    def draw_processing_time(self,img):
        cv.putText(img,self.finish_time,(10,30),cv.FONT_HERSHEY_COMPLEX,0.6,(0,255,0))
        return img
        return img
    def show_dices_and_pips(self):
        res = self.kmeans(self.cnt[0][0], 3)
        # if(len(self.cnt[0])<6):
        #     res = self.kn(self.cnt[0],1)
        # else:
        index = 0
        res = res.tolist()
        dice_1 = res.count(0)
        dice_2 = res.count(1)
        dice_3 = res.count(2)
        print('第一顆骰子為：' + str(dice_1) + '點')
        print('第二顆骰子為：' + str(dice_2) + '點')
        print('第三顆骰子為：' + str(dice_3) + '點')
        print('總計有：' + str(dice_1 + dice_2 + dice_3) + '點')
        
    def start(self):
        for img in self._img_list:
            self.draw_pips(img)
            self.show_dices_and_pips()
            cv.waitKey(0)
            cv.destroyAllWindows()
    
if __name__ == '__main__':
    video_capturer = VideoCapturer(0)
    image_pretreatmenter = ImagePretreatmenter(video_capturer.frames)
    pattern_matcher = PatternMatcher(image_pretreatmenter)
    
    