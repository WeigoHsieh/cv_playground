# import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import cv2 as cv
# import tensorflow.examples.tutorials.mnist
import time

WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + 'dice_groups/'
DICE_WRITE_DIR = WORK_DIR + 'diceing/5/'


class VideoCapturer:
    def __init__(self, camera):
        self.frame = []
        self.start(camera)

    def start(self, camera):
        cap = cv.VideoCapture(camera, cv.CAP_DSHOW)
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
        print('花費了：' + str((end - start)*1000) + '毫秒')

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
        return cv.medianBlur(img, 5)

    def diceblock(self, img):
        blur = self.medianBlur(img)
        red = cv.split(blur)[2]
        dice_blocks = cv.threshold(red, 209, 255, 1)  # 185 --> 235
        in_block = 255 - dice_blocks[1]
        return in_block


    def adaptiveGaussianThresHolding(self, img):
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # 灰階

    def gray(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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
        return cv.Canny(img, 50, 95)

    # 高斯模糊
    def GaussianBlur(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)

    # 拉普拉斯算子
    def lapl(self, img):
        return cv.Laplacian(img, cv.CV_64F)

    # 銳利
    def curve(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]], np.float32)
        dst = cv.filter2D(img, -1, kernel=kernel)
        return dst

    # 霍夫曼圓形檢測(灰值)
    def houghCircle(self, img):
        gray = self.gray(img)
        gau = self.GaussianBlur(gray)
        canny = self.canny(gau)
        circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 100, param1=10,
                          param2=15, minRadius=10, maxRadius=30)  # 把半徑範圍縮小點，檢測內圓，瞳孔
        print(len(circles[0]))
        for circle in circles[0]:
        
            x = np.int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            img = cv.circle(img,(x,y),r,(0,255,0),-1)
            
            return img
            
    def pip_contourfy(self):
        dice_1 = cv.imread(TEMPLATE_GROUP + 'dice_1.png')
        gray = self.gray(dice_1)
        gau = self.GaussianBlur(gray)
        template = self.canny(gau)
        cnts, _ = cv.findContours(
            template, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        return cnts

    # 輪廓檢測(需要灰、模糊、或者二值化)
    def contours(self, img):
        clone = img.copy()
        gray = self.gray(img)
        gau = self.GaussianBlur(gray)
        hou = self.houghCircle(img)
        # canny = self.canny(gau)
        # # hou = self.houghCircle(img)
        # sobel = self.sobel(gray)
        # # method = cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
        # cnts, _ = cv.findContours(
        #     hou, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
     
        # for i, cnt in enumerate(cnts):
        #     print(i)
        #     print(len(cnt))
        #     ret = cv.matchShapes(cnt, self.pip_contourfy()[i+1], 1, 0.0)
        #     # if ret < 0.1:
        #     # print(ret)

        
        # cv.drawContours(clone, cnts, -1, (0, 255, 0), 2)
        # count = 0
        # ares_avrg = 0
        # for cont in cnts:
        #     arc = cv.arcLength(cont, True)
        #     ares = cv.contourArea(cont)  # 計算包圍性狀的面積
        #     self.ares.append(arc)
        #     if ares < 55:  # 過濾面積小於50的形狀
        #         continue
        #     count += 1  # 總體計數加1
        #     ares_avrg += ares

        # print('總共有：' + str(count) + '點')
        # print('---')
        # print(len(self.ares))
        # # print('共有:' + str(hou))
        return hou

    def processing(self):
     
        for i in self._img_list:
            cont = self.contours(i)
            gray = self.gray(i)
            med = self.GaussianBlur(gray)
            sobel = self.sobel(med)
            yield cont

    # 圖片裁減
    def cut(self, img, x, y, w, h):
        crop_img = img[y:y+h, x:x+w]
        return crop_img

    def blob_dection(self, img):
        detector = cv.SimpleBlobDetector()
        keypoints = detector.detect(img)
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array(
            []), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
    # template_preload_saobel()

    # video = VideoCapturer(0)
    # imagePretreatmenter = ImagePretreatmenter(video.frame)
    # patternMatcher = PatternMatcherTest(WORK_DIR + './blob.png', WORK_DIR + 'origin.png')
    cv.waitKey(0)
    cv.destroyAllWindows()
    # a = cv.imread(TEMPLATE_GROUP + './dice_ref.png')
    # cv.imshow('',a)
    # cv2.rectangle(img, 左上(20, 60), 右下(120, 160), 顏色(0, 255, 0), 粗細2)
