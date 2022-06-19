import cv2
import numpy as np
from reinhard import reinhard
import math
import os

class contours:
    def __init__(self):
        self.left = np.array((1,1))
        self.Right =np.array((1,1))
        self.top = np.array((1,1))
        self.Bot = np.array((1,1))


    def normalise(self,target_path,image):
        target = np.load(target_path, mmap_mode='r')
        img = reinhard(image, target)
        return img

    def processing(self,normalized_img):
        kernel2 = np.ones((5, 5), np.float32) / 25
        img = cv2.filter2D(src=normalized_img, ddepth=-1, kernel=kernel2)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        return im

    def detection(self,processed_img):
        lower_blue = np.array([20, 135, 105])
        upper_blue = np.array([115, 190, 190])
        mask_blue = cv2.inRange(processed_img, lower_blue, upper_blue)
        contours, hierarchy = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        con_list = []
        c = 0
        b0 = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area < 15:
                continue
            con_list.append(i)
            c = c+1
        return con_list

    def sorting(self,click,contours):
        out = np.zeros((1,1))
        bo = 0
        for c in contours:
            dist = cv2.pointPolygonTest(c, click, True)
            if dist >= 0:
                out = c
                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                distfromtop = math.sqrt(((extTop[0] - click[0]) ** 2) + ((extTop[1] - click[1]) ** 2))
                distfromLeft = math.sqrt(((extLeft[0] - click[0]) ** 2) + ((extLeft[1] - click[1]) ** 2))
                distfromBot = math.sqrt(((click[0] - extBot[0]) ** 2) + ((click[1] - extBot[1]) ** 2))
                distfromRight = math.sqrt(((click[0] - extRight[0]) ** 2) + ((click[1] - extRight[1]) ** 2))
                condition = np.array([distfromtop, distfromLeft, distfromBot, distfromRight])

                if any(t > 25 for t in condition):
                    if distfromtop>25:

                        if extTop[0] > click[0]:
                           self.top[0] = click[0] - 20
                        elif extTop[0] < click[0]:
                            self.top[0] = click[0] + 20
                        elif extTop[1] > click[1]:
                            self.top[1] = click[1] - 20
                        elif extTop[1] < click[1]:
                            self.top.y = click[1] + 20
                    else:
                        self.top[0] = extTop[0]
                        self.top[1] = extTop[1]

                    if distfromLeft >25:
                             if extLeft[0] > click[0]:
                                 self.left[0] = click[0] - 20
                             elif extLeft[0] < click[0]:
                                 self.left[0] = click[0] + 20
                             elif extLeft[1] > click[1]:
                                 self.left[1] = click[1] - 20
                             elif extLeft[1] < click[1]:
                                 self.left[1] = click[1] + 20
                    else:
                             self.left[0] = extLeft[0]
                             self.left[1] = extLeft[1]

                    if distfromBot > 25:
                         if extBot[0] > click[0]:
                             self.Bot[0] = click[0] - 20
                         elif extBot[0] < click[0]:
                             self.Bot[0] = click[0] + 20
                         elif extBot[1] > click[1]:
                             self.Bot[1] = click[1] - 20
                         elif extBot[1] < click[1]:
                             self.Bot[1] = click[1] + 20
                    else:
                        self.Bot[0] = extBot[0]
                        self.Bot[1] = extBot[1]

                    if distfromRight >25:
                        if extRight[0] > click[0]:
                           self.Right[0] = click[0] - 20
                        elif extRight[0] < click[0]:
                            self.Right[0] = click[0] + 20
                        elif extRight[1] > click[1]:
                            self.Right[1] = click[1] - 20
                        elif extRight[1] < click[1]:
                            self.Right[1] = click[1] + 20
                    else:
                        self.Right[0] = extRight[0]
                        self.Right[1] = extRight[1]
                    con = [[self.top],[self.Right],[self.Bot],[self.left]]
                    out = np.array(con)
                    bo = 1
            else:
                continue
        if out.all() == 0:
            print("point outside nuclei")

        return out,bo


def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        click = (x,y)
        final_contour,bol = contour_detection.sorting(click,detected_list)

        if bol:
            cv2.ellipse(image, click, (4,7),
                        0, 0, 360,(0, 255, 0),1)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        elif final_contour.all() :
           cv2.drawContours(image, final_contour, -1, (0, 255, 0), 1)
           cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

path = os.getcwd()
path_img = path+'/input/test_img.png'
if not os.path.exists(path_img):
    path_img = input("path_img")
image = cv2.imread(path_img)
target_path = path + '/target.npy'
contour_detection = contours()
norm = contour_detection.normalise(target_path, image)
process = contour_detection.processing(norm)
detected_list = contour_detection.detection(process)
cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB',mouseRGB)

#Do until esc pressed
while(1):
    cv2.imshow('mouseRGB',image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
#if esc pressed, finish.
cv2.destroyAllWindows()