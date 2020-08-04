# import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv


WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"
TEMPLATE_GROUP = WORK_DIR + './dice_groups/'

class Assignment:
    def __init__(self,img_file_path):
        self.img_file_path = img_file_path
        self.img_rgb = self.img_init()
        self.img_gray = self.get_gray_img(self.img_rgb)
        self.video_capture()
       
       
    def video_capture(self):
        cap = cv.VideoCapture(0) 
        fourcc = cv.VideoWriter_fourcc(*'XVID') 
        out = cv.VideoWriter('./output.avi',fourcc, 20.0, (640,480)) 
        while(True): 
            ret, frame = cap.read() 
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            out.write(frame) 
            cv.imshow('frame',gray) 
            if cv.waitKey(1) & 0xFF == ord('q'): 
                break 
            cap.release() 
            out.release() 
            cv.destroyAllWindows()

    def get_gray_img(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

    def img_init(self):
        return cv.imread(self.img_file_path)

    def get_template(self, string_number):
        template =  cv.imread(TEMPLATE_GROUP + './dice_' + string_number + '.png', 0)
        h =  template.shape[::-1]
        res = cv.matchTemplate(self.to_img_gray(),template,cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res>=threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(self.img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv.imwrite('res.png',self.img)

    def preload_image(self):
        self.img = cv.imread(self.img_file_path)
        self.show(self.img)
        
    def to_canny (self):
        return cv.Canny(self.img, 100,200)
    
    
    def show (self,show):
        plt.imshow(show)
        
        
    def to_img_gray(self):
        return cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)


if __name__ == "__main__":
    a = Assignment(WORK_DIR + './diss.png')
    