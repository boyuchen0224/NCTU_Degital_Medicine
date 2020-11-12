import os
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('..\\Train_png_all\\aug_train_img.csv',  header=None)
        label = pd.read_csv('..\\Train_png_all\\aug_train_label.csv',  header=None)
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('..\\Train_png_all\\test_img.csv', header=None)
        label = pd.read_csv('..\\Train_png_all\\test_label.csv', header=None)
        return np.squeeze(img.values), np.squeeze(label.values)


class HemorrhageLoader(data.Dataset):
    def __init__(self, root, mode, transform_func):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status("train" or "Val")
            transform_func: how to transfer the image to tensor

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transform_func
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """
           step1. Get the image path from 'self.img_name' and load it.
           step2. Get the ground truth label from self.label
           step3. Transpose the image shape from [H, W, C] to [C, H, W]                         
           step4. Return processed image and label
        """
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root, self.img_name[index])

        raw_img = Image.open(img_name)
        rgb_img = raw_img.convert('RGB')
        img = self.transform(rgb_img)        
        label = self.label[index]

        return {'img': img, 'label': label}
    
def transform_func():
    # return transforms.Compose([
    #     transforms.Resize((512,512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])

    # # efficientnet-b0 model 001 and 002
    # return transforms.Compose([
    #     transforms.CenterCrop((512,512)),
    #     transforms.ToTensor()
    # ])
     
    # # efficientnet-b0 model 003
    return transforms.Compose([
        transforms.CenterCrop((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])    

if __name__ == '__main__':
    root, mode = "..\\Train_png_all\\photo\\", "train"
    data_transform = transform_func()

    batch_size = 1
    dataset = HemorrhageLoader(root, mode, data_transform)
    data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle=True, num_workers = 0)

    # example of data loading
    for i_batch, sampled_batch in enumerate(data_loader):
        inputs = sampled_batch['img']
        labels = sampled_batch['label']

        # # *** show example of input images ***
        # a = inputs[0].numpy().transpose((1, 2, 0))
        # plt.figure()
        # plt.imshow(a)
        # plt.title('class: ' + str(labels[0].numpy()))
        # plt.show()

        print(inputs.shape)
        # if i_batch > 3:
        #     break
