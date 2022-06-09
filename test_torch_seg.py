import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


best_model = torch.load('./mobilenet_v2_best_model.pt')
DATA_DIR = '/home/compu/jh/project/yolov5/'
img_path = "./sesame_origin/test/seg_img_png/20210814_132321_jpg.rf.aa779ce4cfc2c332d997db5275227df0.png"
image = cv2.imread(img_path)
image = cv2.resize(image, (512, 512))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pt_model = torch.load('./timm-mobilenetv3_small_100_best_model1.pt', map_location='cpu')

transform = albu.Compose([
    albu.Resize(512, 512),
    albu.Normalize(),
    ToTensorV2(),
])
transformed_image = transform(image=image)['image']
transformed_image = torch.unsqueeze(transformed_image, 0)

torch_out = pt_model(transformed_image)

visualize(pt_model(transformed_image).squeeze().detach().cpu().numpy().round())
