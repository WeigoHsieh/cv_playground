# import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv


WORK_DIR = "C:/Users/09080381/Desktop/assignment/1.dice_classification/"

class Assignment:
    def __init__(self,img_file_path):
        self.img_file_path = img_file_path
        self.preload_image()
        self.to_img_gray()
        self.show()
    
    def preload_image(self):
        self.img = cv.imread(self.img_file_path,cv.IMREAD_GRAYSCALE)
    
    
    def show (self):
        plt.imshow(self.img, interpolation = "bicubic")
        
        
    def to_img_gray(self):
        cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)




if __name__ == "__main__":
    a = Assignment(WORK_DIR + './diss.png')
    