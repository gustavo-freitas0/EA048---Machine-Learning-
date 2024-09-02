import fastapi
import cv2 as cv
import numpy as np
from fastapi import FastAPI
import matplotlib.pyplot as plt

'''

    This code is for neural net API
    
    First we are work with API (without net API)
    
    Divided into two classes
    
    * TakeImage
    - Open the webcam 
    - Take a frame 
    - identify a digit (we need to do something about canny edge)
    - crop the image
    - insert the image in model
    
    OR 
    
    * UploadImage
    - Just insert a image in the code
    - identify a digit (we need to do something about canny edge)
    - crop the image
    - insert the image in model
    
    

'''


class TakeImage:

    def __init__(self):
        print('Here we need take a picture and process the image to appy the model')

        self.image = self.take_photo()
        self.crop_image(self.image)

    def take_photo(self) -> np.ndarray:
        print('Opening the webcam')
        webcam_ = cv.VideoCapture(0)
        image_ = None
        if not webcam_.isOpened():
            print('It was not possible to open the webcam')
            exit()
        while True:
            ret, frame = webcam_.read()
            if not ret:
                print('The capture failed')
                break
            cv.imshow('WebCam Image', frame)
            if cv.waitKey(1) == 32:
                image_ = frame
                print('Image captured')
                break
        webcam_.release()
        cv.destroyAllWindows()
        return image_

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        '''
        Here I thought just crop the image in the center, like QR code identify
        :param image: Webcam image capture
        :return: Image processed and resized
        '''
        print(f'image shape is {image.shape}')
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_invert = cv.bitwise_not(image_gray)
        image_gaussian_blur = cv.GaussianBlur(image_invert, (3, 3), 0)
        image_bilateral_blur = cv.bilateralFilter(image_gaussian_blur, 1, 25, 25)
        image_canny = cv.Canny(image_bilateral_blur, 100, 150)

        image_crop = np.copy(image_canny)
        height, width = image_crop.shape
        image_crop = image_crop[int(0.2 * height):int(0.8 * height), int(0.2 * width):int(0.8 * width)]
        cv.imshow('Crop image', image_crop)
        if cv.waitKey(0) == ord('q'):
            cv.destroyAllWindows()
        return cv.resize(image_crop, (28, 28))


class UploadImage:

    def __init__(self):
        print('I dont know what do here')


if __name__ == "__main__":
    print("Code inited")
    collect_image = TakeImage()
