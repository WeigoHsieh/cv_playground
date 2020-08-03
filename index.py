import cv2 
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./diss2.png').astype(np.float32) / 255


img = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)


GAMMA = 0.5

corrected_img = np.power(img,GAMMA)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

plt.imshow(th2,'')