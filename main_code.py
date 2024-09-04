import fastapi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2 as cv
import os
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
        # print('Here we need take a picture and process the image to appy the model')

        self.image = self.take_photo_with_target()
        final_image_ = self.crop_image(self.image)

        self.pred_, self.prob = self.apply_model(final_image_)

        cv.putText(self.image, f'{int(self.pred_)} - {self.prob:.4f}', (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv.imshow('Prediction', self.image)
        if cv.waitKey(0) == ord('f'):
            pass
        print(f'The final result is: {self.pred_} with {self.prob:.4f}')

    @staticmethod
    def take_photo_with_target() -> np.ndarray:
        # print('Opening the webcam')
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
            h_, w_, _ = frame.shape
            frame_with_rect = cv.rectangle(frame, (int(0.2 * w_), int(0.2 * h_)), (int(0.8 * w_), int(0.8 * h_)),
                                           (0, 255, 0), 2)
            cv.imshow('WebCam Image', frame_with_rect)
            if cv.waitKey(1) == 32:
                image_ = frame
                # print('Image captured')
                break
        webcam_.release()
        cv.destroyAllWindows()
        return image_

    @staticmethod
    def crop_image(image: np.ndarray) -> np.ndarray:
        '''
        Here I thought just crop the image in the center, like QR code identify
        :param image: Webcam image capture
        :return: Image processed and resized
        '''
        # print(f'image shape is {image.shape}')
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_invert = cv.bitwise_not(image_gray)
        image_gaussian_blur = cv.GaussianBlur(image_invert, (3, 3), 0)
        image_bilateral_blur = cv.bilateralFilter(image_gaussian_blur, 1, 25, 25)
        image_canny = cv.Canny(image_bilateral_blur, 100, 150)

        image_crop = np.copy(image_canny)
        height, width = image_crop.shape
        image_crop = image_crop[int(0.2 * height):int(0.8 * height), int(0.2 * width):int(0.8 * width)]
        # cv.imshow('Crop image', image_crop)
        # if cv.waitKey(0) == ord('q'):
        #     cv.destroyAllWindows()
        return cv.resize(image_crop, (28, 28))

    @staticmethod
    def find_weights() -> str:
        weight_file = None
        for _, _, files in os.walk(os.getcwd()):
            files_pt = [file for file in files if file.endswith('.pt')]
            if len(files_pt) != 0:
                weight_file = files_pt[0]
                return weight_file
        return weight_file

    def apply_model(self, image_data: np.ndarray) -> tuple:

        pt_file_ = r'C:\Users\Gustavo - Win\Documents\AI_projects\IA048_Machine_Learning\best_model_model_1541.pt'

        cnn = CNN()

        state_dict_ = torch.load(pt_file_, map_location='cpu', weights_only=True)

        cnn.load_state_dict(state_dict_)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        data_tensor_ = transform(image_data)
        data_batch_ = data_tensor_.unsqueeze(0)

        cnn.eval()

        with torch.no_grad():
            out_ = cnn(data_batch_)

        _, pred_ = torch.max(out_, 1)

        # print(f'{pred_=}')

        probs = torch.nn.functional.softmax(out_[0], dim=0)

        # print(f'{probs=}')

        pred_prob = probs[pred_[0]].item()

        # print(f'{pred_prob=}')

        return pred_.item(), pred_prob


class UploadImage:

    def __init__(self):
        print('I dont know what do here')


class CNN(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(CNN, self).__init__()
        # First convolucional layer with Batch normalization and Dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=0.25)

        # Second convolucional layer with Batch normalization and Dropout
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(p=0.25)

        # Polling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Fully connected layer
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Convolucion with ReLU, batch normalization and dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        # Convolucion with ReLU, batch normalization and dropout
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        # Dimension reduction (reduz consumo computacional)
        x = self.pool(x)

        # Flattening
        x = x.view(x.size(0), -1)

        # Fully connected layer with ReLU and Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop3(x)

        # Last layer
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    print("Code inited")
    collect_image = TakeImage()
