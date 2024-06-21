import numpy as np
import cv2
import os
import shutil
import matplotlib
from matplotlib import pyplot as plt

def get_cropped_image_2_eyes(img_path):
    
    face_detector = cv2.CascadeClassifier('E:\LAB\haarcascade_frontalface_default.xml')
    eye_detector = cv2.CascadeClassifier("E:\LAB\haarcascade_eye.xml")
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


if __name__ == "__main__":

    path_to_data='E:\LAB\Celebrity Faces Dataset'
    percorso_directory ='E:/LAB/cropped/'
    os.makedirs(percorso_directory, exist_ok=True)
    path_to_cropped='E:/LAB/cropped/'
    img_dirs =[]
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
            img_dirs.append(entry.path)
    if os.path.exists(path_to_cropped):
        shutil.rmtree(path_to_cropped)
    os.mkdir(path_to_cropped)

    cropped_img_dirs =[]
    celeb_file_names_dict ={}

    for img_dir in img_dirs:
        count = 1
        celeb_name = img_dir.split('\\')[-1]

        celeb_file_names_dict[celeb_name]=[]

        print(celeb_name)
        for entry in os.scandir(img_dir):
            roi_color = get_cropped_image_2_eyes(entry.path)
            if roi_color is not None:
                cropped_folder = path_to_cropped + celeb_name
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_img_dirs.append(cropped_folder)
                    print("generating in folder", cropped_folder)
                cropped_file_name = celeb_name + str(count) +".jpg"
                cropped_file_path = cropped_folder +"\\" + cropped_file_name
            
                cv2.imwrite(cropped_file_path,roi_color)

                celeb_file_names_dict[celeb_name].append(cropped_file_path)
                count +=1
