from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import pydicom
import numpy as np
import cv2
from skimage.measure import label 
import matplotlib.pyplot as plt

class DICOMFolder(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.dicom_files, self.labels = self._find_dicom_files_and_labels()

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_file = self.dicom_files[idx]
        dicom = pydicom.dcmread(dicom_file)

        if 'CT' in dicom_file:
            #print(dicom_file)
            process_ds = self._preprocess_cbct_ct(dicom)

        if 'MRI' in dicom_file:
            #print(dicom_file)
            process_ds = self._preprocess_mri(dicom)

        pixel_array_stretched = (process_ds * 255).astype(np.uint8)

        # Apply transformations
        if self.transform:
            PIL_array = Image.fromarray(pixel_array_stretched, mode='L').convert('RGB')
            pixel_array = self.transform(PIL_array)

        # Convert to tensor and return along with the label
        label = self.labels[idx]
        image_sample = torch.tensor(pixel_array, dtype=torch.float32)
        label_sample = torch.tensor(label, dtype=torch.long)

        return image_sample, label_sample

    def _find_dicom_files_and_labels(self):
        dicom_files = []
        labels = []
        label_map = {}  # To store a mapping of subdirectory names to label indices
        label_idx = 0

        for class_dir in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, class_dir)):
                class_path = os.path.join(self.root, class_dir)
                label_map[class_dir] = label_idx
                label_idx += 1

                for file in os.listdir(class_path):
                    if file.lower().endswith('.dcm'):
                        dicom_files.append(os.path.join(class_path, file))
                        labels.append(label_map[class_dir])
        
        return dicom_files, labels
    
    def getLargestCC(self,segmentation):
            labels = label(segmentation)
            assert( labels.max() != 0 ) # assume at least 1 CC
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
            return largestCC
    
    def del_coach(self,img):
        mm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        im = np.array(mm, dtype='uint8')
        ret, thresh1 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        mcc = self.getLargestCC(thresh1) 
        fil = img * mcc
        kernel = np.ones((3,3), np.uint8)
        ero = cv2.erode(fil, kernel, iterations=1)
        return ero
    
    def _preprocess_cbct_ct(self,dicom):
        hu_data = dicom.pixel_array.astype(np.float32) * dicom.RescaleSlope + dicom.RescaleIntercept
        clip_hu = np.clip(hu_data, -1000, 1000)
        nor_hu = (clip_hu + 1000.) / (1000. + 1000.)
        process_data = self.del_coach(nor_hu)
        return process_data
    
    def _preprocess_mri(self,dicom):
        pixel_array = dicom.pixel_array.astype(np.float32) * dicom.RescaleSlope + dicom.RescaleIntercept
        clip_in = np.clip(pixel_array, 0, 1500)
        nor_hu = clip_in / (1500)
        return nor_hu

    
def get_loader(train_dir, test_dir, validate_dir, image_size=256, 
               batch_size=16, mode='train', num_workers=1, augment=False):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    if augment:
        transform.append(T.RandomRotation(degrees = (-30,30)))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Lambda(lambda x: x.expand(3, -1, -1)))
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if mode == 'train':
        dataset = DICOMFolder(train_dir, transform)
        val_dataset = DICOMFolder(validate_dir, transform)
    elif mode == 'test':
        dataset = DICOMFolder(test_dir,transform)
        val_dataset = DICOMFolder(validate_dir, transform)
        batch_size = 16

    data_loader = data.DataLoader(dataset=dataset,                 
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    
    val_data_loader = data.DataLoader(dataset=val_dataset,                 
                                  batch_size=16,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader, val_data_loader
    
    
def get_loader_class(train_dir, image_size=128, 
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader for training the classifier with data augmentation"""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip(p=0.5))
    transform.append(T.Resize(image_size))
    transform.append(T.RandomRotation(degrees = (-20,20)))
    transform.append(T.ToTensor())
    transform.append(T.Lambda(lambda x: x.expand(3, -1, -1)))
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = DICOMFolder(train_dir, transform)
         
    data_loader = data.DataLoader(dataset=dataset,                  
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
