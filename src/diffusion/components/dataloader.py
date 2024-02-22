# Bismillah Hirrahman Nirrahim 

import os 
import numpy as np 
import torch 
import albumentations as A 
from PIL import Image

transform = A.Compose([
    A.Resize(256 , 256),
    A.Normalize(mean = (0.485 , 0.456 , 0.406) , std = (0.229 , 0.224 , 0.225)),
])


class DiffusionDataset(torch.utils.data.Dataset):
    """
    The DiffusionDataset class is a subclass of dataset.Dataset. It is used to handle and manipulate 
    image datasets stored in a directory for diffusion processes.

    Attributes:
        data_dir (str): The directory where the image dataset is stored.
        transform (callable, optional): Optional transform to be applied on an image.
        images (list): List of names of the image files in the data directory.

    Methods:
        __len__: Returns the number of images in the dataset.
        __getitem__: Returns the image at the specified index after applying the transform if any.
    """

    def __init__(self , data_dir , transform = None):
        """
        The constructor for DiffusionDataset class.

        Parameters:
            data_dir (str): The directory where the image dataset is stored.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir 
        self.transform = transform 
        self.images = os.listdir(data_dir)

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.images)

    def __getitem__(self , idx):
        """
        Returns the image at the specified index after applying the transform if any.

        Parameters:
            idx (int): The index of the image.

        Returns:
            np.array: The image at the specified index.
        """
        img_path = os.path.join(self.data_dir , self.images[idx])
        image = np.array(Image.open(img_path))
        if self.transform: 
            image = self.transform(image = image)['image']
        return image