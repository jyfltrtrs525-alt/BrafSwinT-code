import os
from os import path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# root path
str1 = input("Which part of the data do you want to be labeled: ")
print("The data to be labeled are: Part", str1)

root_path = r'C:\python files\bz\part'+str1

file = os.listdir(root_path)
for f in file:
    patient_path = path.join(root_path, f)
    for i in range(1,3):
        img_name = str(i) + '.jpg'
        img_path = path.join(patient_path, img_name)
        img1 = cv2.imread(img_path)
        img = np.array(img1)
        try:
            cv2.namedWindow('ROI', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('ROI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            ROI = cv2.selectROI('ROI',img,False,False)
        except:
            exit()
            cv2.destroyAllWindows()

        x,y,w,h = ROI

        'expand 3mm'
        try:
            cv2.namedWindow('LROI', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('LROI', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            LROI = cv2.selectROI('LROI',img,False,False)
            cv2.destroyAllWindows()
        except:
            exit()
            cv2.destroyAllWindows()
        _, _, _, l = LROI

        dt = int(l/10*3)  # 2cm/20*3
        x1 = int(x - dt)
        x2 = int(x + w + dt)
        y1 = int(y - dt)
        y2 = int(y + h + dt)

        roi_img = img[y1:y2,x1:x2,:]
        kong = np.zeros((roi_img.shape[0], roi_img.shape[1], 3), np.uint8)
        kong[:,:,:] = roi_img[:,:,:]

        save_path = patient_path + '\\ROI'+str(i) + '.jpg'
        cv2.imwrite(save_path, roi_img)
    continue

print("Finished!)")