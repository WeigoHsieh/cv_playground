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
        return cv.medianBlur(img, 7)

    def diceblock(self, img):
        blur = self.medianBlur(img)
        red = cv.split(blur)[2]
        dice_blocks = cv.threshold(red, 209, 255, 1)  # 185 --> 235
        in_block = 255 - dice_blocks[1]
        return in_block


    def adaptiveGaussianThresHolding(self, img):
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # 灰階
    
    
    def kn(self,data):
        km = KMeans(n_clusters=3,
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
        print(km)
        result = km.fit_predict(data)
        return result

        
        
        

    def gray(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def ex(self,img):
        kernel = np.ones((5,5),np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return opening
    
    
    def pip_distance(self,x,y):
        distance = math.sqrt(math.pow(x,2) + math.pow(y,2))
        return distance
    
    
    # 索伯
    def sobel(self, img):
        x = cv.Sobel(img, cv.CV_8U, 1, 0)
        y = cv.Sobel(img, cv.CV_8U, 0, 1)

        absX = cv.convertScaleAbs(x)  # 轉回uint8
        absY = cv.convertScaleAbs(y)

        dst = cv.addWeighted(absX, 1, absY, 1, 0)
        return dst

    # 邊緣檢測
    def canny(self, img):
        return cv.Canny(img, 120, 350) 
    # 120,350 200,400

    # 高斯模糊
    def GaussianBlur(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)


    # 霍夫曼圓形檢測(灰值)
    def houghCircle(self, img):
        gray = self.gray(img)
        gau = self.GaussianBlur(gray)
        med = self.medianBlur(gray)
        canny = self.canny(self.ex(img))
        
        cnt = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 10, param1=10,
                          param2=15, minRadius=5, maxRadius=30)  # 把半徑範圍縮小點，檢測內圓，瞳孔
        total = 0
        if(cnt is None):
            print('找不到')
            return
        else:
            for circles in cnt:
                for cp in circles:
                    r = int(cp[2])
                    if (r < 16 and r > 5):
                        x = np.int(cp[0])
                        y = int(cp[1])
                        # print(self.pip_distance(x,y))
                        img = cv.circle(img,(x,y),r,(0,255,0),-1)
                        total += 1
                    else:
                        pass
        
        res = self.kn(cnt[0])
        res = res.tolist()
        dice_1 = res.count(0)
        dice_2 = res.count(1)
        dice_3 = res.count(2)
        print('第一顆骰子為：' + str(dice_1) + '點')
        print('第二顆骰子為：' + str(dice_2) + '點')
        print('第三顆骰子為：' + str(dice_3) + '點')
        print('總計有：' + str(dice_1 + dice_2 + dice_3) + '點')
        return img
    
    # def list_count(self,dataframe,number):
    #     pass
            
    # def pip_contourfy(self):
    #     dice_1 = cv.imread(TEMPLATE_GROUP + 'dice_1.png')
    #     gray = self.gray(dice_1)
    #     gau = self.GaussianBlur(gray)
    #     template = self.canny(gau)
    #     cnts, _ = cv.findContours(
    #         template, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #     return cnts

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

    # 圖片裁減
    # def cut(self, img, x, y, w, h):
    #     crop_img = img[y:y+h, x:x+w]
    #     return crop_img

    # def blob_dection(self, img):
    #     detector = cv.SimpleBlobDetector()
    #     keypoints = detector.detect(img)
    #     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #     img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array(
    #         []), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     return img_with_keypoints


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
    def __init__(self, templ, origin):
        self.templ = templ
        self.origin = origin
        self.method = cv.TM_CCOEFF_NORMED

    def templateMatching(self):
        res = self.match(self.method)
        return res

    def match(self, method):
        return cv.matchTemplate(self.gray, self.templ, method)

    def detect(self):
        pass

    def monent(self, contours):
        for i in contours.size():
            yield cv.monents(contours[i], False)

    def shapeMatching(self, origin, templ):
        return cv.matchShapes(origin, templ, 1, 0.0)

    def test(self):
        self.templ = cv.imread(WORK_DIR + 'blob.png')
        # 化緣
        mark(self.origin, self.templ)

    def mark(self, origin_gray, templ):
        # //matchTemplate會回傳座標
        # return
        w, h = templ.shape[::-1]
        res = cv.matchTemplate(origin_gray, templ, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= self.threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(self.origin_gray, pt,
                         (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


class PatternMatcherTest:
    def __init__(self, template, origin_img):
        self.template = cv.imread(template)
        self.origin_img = cv.imread(origin_img)
        self.gray = cv.cvtColor(self.origin_img, cv.COLOR_RGB2GRAY)
        self.threshold = 0.8
        self.start()

    def blob_dection(self, img):
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
        self.detect(self.gray, self.template)
        print(self.template, self.gray)
        # cv.imshow('',self.origin_img)

    def detect(self, origin_gray, templ):
        pass
        # //matchTemplate會回傳座標
        # return
        # w,h = templ.shape[::-1]
        # res = cv.matchTemplate(origin_gray, templ, cv.TM_CCOEFF_NORMED)
        # loc = np.where (res >= self.threshold)
        # for pt in zip(*loc[::-1]):
        #     cv.rectangle(self.origin_gray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


class DiceImageTrainer:
    def __init__(self, pretreatmenter):
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
            TEMPLATE_GROUP_DIR = TEMPLATE_GROUP + 'dice_' + str(i+1) + '.png'
            print(TEMPLATE_GROUP_DIR, '\n')
            img = cv.imread(TEMPLATE_GROUP)
            # cv.imshow('',img)
        # destory()

    preload_test()
    
    def kn(self,data):
        num_clusters = 3
        km_cluster = KMeans(n_clusters=num_clusters, max_iter=150, n_init=40, \
                    init='k-means++')
        result = km_cluster.fit_predict(data)
        print(result)
    
        
    # data = [[342.5, 313.5 , 10.6],
    #         [381.5, 291.5 , 11.8],
    #         [296.5, 163.5 , 12.5],
    #         [281.5, 213.5 , 11.4],
    #         [317.5, 273.5 , 12.2],
    #         [270.5, 169.5 , 11.4],
    #         [320.5, 156.5 , 11.8],
    #         [202.5, 186.5 , 11.4],
    #         [331.5, 199.5 , 11.8],
    #         [357.5, 250.5 , 11.8],
    #         [305.5, 207.5 , 11.4]]
    # kn(data)
    
    # template_preload_saobel()

    # video = VideoCapturer(0)
    # imagePretreatmenter = ImagePretreatmenter(video.frame)
    # patternMatcher = PatternMatcherTest(WORK_DIR + './blob.png', WORK_DIR + 'origin.png')
    cv.waitKey(0)
    cv.destroyAllWindows()
    # a = cv.imread(TEMPLATE_GROUP + './dice_ref.png')
    # cv.imshow('',a)
    # cv2.rectangle(img, 左上(20, 60), 右下(120, 160), 顏色(0, 255, 0), 粗細2)
