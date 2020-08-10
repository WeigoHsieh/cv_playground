流程步驟：
- [x] I.找到攝影機位置並拍照。
- [x] II.圖像預處理
- [] III.判定點數


# I.找到攝影機位置並拍照。


# II.圖像預處理

1.背景加入
2.灰階
3.邊緣檢測
4.ad

# III.判定點數
[] 1.分割判斷
[x] 2.點數加總

### 先利用邊緣檢測template matching判斷是不是骰子，判斷OK就在template裡面做hough circle。


https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python