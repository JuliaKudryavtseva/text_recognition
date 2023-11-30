from PIL import Image
import cv2
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2

import warnings
warnings.filterwarnings('ignore')


class TextRecognitionDataset(Dataset):

    def __init__(self, phase, root=r'C:\Users\julia\Documents\masters\deep_learning\text_recognition\data', tranform=None, preprocessing=None):

        self.tranform = tranform
        self.preprocessing =preprocessing
        self.phase = phase

        if phase == 'train':
            self.image_path_simple = os.path.join(root, phase, phase, 'simple')
            self.image_path_complex = os.path.join(root, phase, phase, 'complex')

            self.simple = os.listdir(self.image_path_simple)
            self.complex = os.listdir(self.image_path_complex)

            self.all_images = self.simple + self.complex 
            self.image2label = {label: label.split('.')[0].split('_')[-1] for label in self.all_images}

        elif phase == 'test':
            self.image_path = os.path.join(root, phase, 'result')
            self.all_images = os.listdir(self.image_path)
            self.image2label = {label: label.split('.')[0].split('_')[-1] for label in self.all_images}
            
        else:
            raise ValueError('Wrong name for phase: choose train or test')  
        
    def __getitem__(self, index):

        image = self.all_images[index]
        label = self.image2label[image]

        if self.phase == 'train':

            if image in self.simple:
                image  = Image.open(os.path.join(self.image_path_simple, image))

            else:
                image  = Image.open(os.path.join(self.image_path_complex, image))

        else:
            image  = Image.open(os.path.join(self.image_path, image))

        # make one channel (gray)
        image = image.convert('L')


        if self.preprocessing:
            image = self.preprocessing(image)

        if self.tranform:
            image = self.tranform(image)

        return image, label
        
    def __len__(self):
        return len(self.image2label)



def ada_thr(img):
    img = np.array(img)
    ada_mean_thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    ada_gaus_thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    result = np.concatenate([img[:, :, None], ada_mean_thr[:, :, None], ada_gaus_thr[:, :, None]], axis=2)
    return result




defualt_transform = v2.Compose([
                        v2.Resize((32, 160)), 
                        v2.ToTensor(),
                        v2.Normalize((0.1307,), (0.3081,))])



def get_dataloaders(BATCH_SIZE=16, preprcoessing=False, tranform=defualt_transform):
    if preprcoessing:
        preprocessing_fun = ada_thr
    else:
        preprocessing_fun = None

    train_set = TextRecognitionDataset('train', preprocessing=preprocessing_fun, tranform=tranform)
    test_set = TextRecognitionDataset('test', preprocessing=preprocessing_fun, tranform=tranform)

    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return train, test



if __name__ == '__main__':    
    train_dataloader, test_dataloader = get_dataloaders() 
    
    for (image, label) in train_dataloader:
        print('TRAIN')
        print(label)
        print(image.shape)
        break
    

    for (image, _) in test_dataloader:
        print('TEST')
        print(image.shape)
        break
