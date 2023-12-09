from PIL import Image
import cv2
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import torch

import warnings
warnings.filterwarnings('ignore')


abc = "0123456789ABEKMHOPCTYX"

class TextRecognitionDataset(Dataset):
    """Class for training image-to-text mapping using CTC-Loss."""

    def __init__(self, phase, root='./data',  alphabet=abc, transform=None, preprocessing=None):
        """Constructor for class.
        
        Args:
            - phase:         String: 'train' or 'test'
            - root:          Dir with images
            - alphabet:      String of chars required for predicting.
            - transforms:    Transformation for items, should accept and return dict with keys "image", "seq", "seq_len" & "text".
            - preprocessing: Transformation for Complex images with classical cv methods
        """
        super(TextRecognitionDataset, self).__init__()

        self.transform = transform
        self.phase = phase
        self.root = root
        self.alphabet = alphabet

        self.preprocessing = preprocessing

        self._parse_path()
        
    def __getitem__(self, index):
        """Returns dict with keys "image", "seq", "seq_len" & "text".
        Image is a numpy array, float32, [0, 1].
        Seq is list of integers.
        Seq_len is an integer.
        Text is a string.
        """

        image = self.all_images[index]   
        text = self.image2label[image]

        seq = self.text_to_seq(text)
        seq_len = len(seq)


        if self.phase == 'train':
            if image in self.simple:
                image_path = os.path.join(self.image_path_simple, image)    # train simple phase
            else:
                image_path = os.path.join(self.image_path_complex, image)   # train complex phase
        else: 
            image_path = os.path.join(self.image_path, image)               # test phase


        # read images and make one channel (gray)
        image  = cv2.imread(image_path).astype(np.float32) / 255.
        # image = image.convert('L')


        if self.preprocessing:
            image = self.preprocessing(image)


        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        if self.transform:
            output = self.transform(output)
        return output
        

    def __len__(self):
        return len(self.image2label)
    

    def text_to_seq(self, text):
        """Encode text to sequence of integers.

        Args:
            - String of text.
        Returns:
            List of integers where each number is index of corresponding characted in alphabet + 1.
        """
        
        seq = [self.alphabet.find(c) + 1 for c in text]
        
        return seq


    def _parse_path(self):

        if self.phase == 'train':
            self.image_path_simple = os.path.join(self.root, self.phase, self.phase, 'simple')
            self.image_path_complex = os.path.join(self.root, self.phase, self.phase, 'complex')

            self.simple = os.listdir(self.image_path_simple)
            self.complex = os.listdir(self.image_path_complex)

            self.all_images = self.simple + self.complex 
            self.image2label = {label: label.split('.')[0].split('_')[-1] for label in self.all_images}

        elif self.phase == 'test':
            self.image_path = os.path.join(self.root, self.phase, 'result')
            self.all_images = os.listdir(self.image_path)
            self.image2label = {label: label.split('.')[0].split('_')[-1] for label in self.all_images}
            
        else:
            raise ValueError('Wrong name for phase: choose "train" or "test"')  




# preprocessing: Transformation for Complex images with classical cv methods
def ada_thr(img):
    """Function for preprocessing images.
    Args:
        - img: numpy array.
    Returns:
        concatenated numpy array of images with different treshold methods
    """
    ada_mean_thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    ada_gaus_thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    result = np.concatenate([img[:, :, None], ada_mean_thr[:, :, None], ada_gaus_thr[:, :, None]], axis=2)
    return result


class Resize(object):

    def __init__(self, size=(320, 64)):
        self.size = size

    def __call__(self, item):
        """Apply resizing.
        Args: 
            - item: Dict with keys "image", "seq", "seq_len", "text".
        Returns: 
            Dict with image resized to self.size.
        """
        
        interpolation = cv2.INTER_AREA if self.size[0] < item["image"].shape[1] else cv2.INTER_LINEAR
        item["image"] = cv2.resize(item["image"], self.size, interpolation=interpolation)
        
        return item
    

def collate_fn(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.

    Args:
        - batch: List of dataset __getitem__ return values (dicts).
    Returns:
        Dict with same keys but values are either torch.Tensors of batched images or sequences or so.
    """
    images, seqs, seq_lens, texts = [], [], [], []
    for item in batch:
        images.append(torch.from_numpy(item["image"]).permute(2, 0, 1).float())
        seqs.extend(item["seq"])
        seq_lens.append(item["seq_len"])
        texts.append(item["text"])

    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}
    return batch




# defualt_transform = v2.Compose([
#                         v2.Resize((32, 160)), 
#                         v2.ToTensor(),
#                         v2.Normalize((0.1307,), (0.3081,))])
defualt_transform = Resize(size=(320, 64))


def get_dataloaders(BATCH_SIZE=16, preprcoessing=False, transform=defualt_transform):
    if preprcoessing:
        preprocessing_fun = ada_thr
    else:
        preprocessing_fun = None

    train_set = TextRecognitionDataset('train', preprocessing=preprocessing_fun, transform=transform)
    test_set = TextRecognitionDataset('test', preprocessing=preprocessing_fun, transform=transform)

    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_fn)

    return train, test



if __name__ == '__main__':    
    train_dataloader, test_dataloader = get_dataloaders() 
        
    for batch in train_dataloader:
        print('TRAIN')
        print('image', batch['image'].shape)
        print('seq', batch['seq'])
        print('seq_len', batch['seq_len'])
        print('text', batch['text'])
        print()
        # break

    for batch in test_dataloader:
        print('TEST')
        print('image',batch['image'].shape)
        print('seq', batch['seq'])
        print('seq_len', batch['seq_len'])
        print('tresh', batch['text'])
        print()
        # break
