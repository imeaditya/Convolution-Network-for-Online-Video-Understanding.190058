
import os
import csv
 
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

def make_datapath_list (root_path):
    """
    Create a list of paths to the directory where the video is the image data.
    
    Inputs
    ----------
    root_path: str
        Root path to the data directory
    
    Returns
    ----------
    video_list: list (str)
        List of paths to directories with video as image data
    """

    video_list = list ()
    class_list = os.listdir (path = root_path)

    # Get the path to the directory where the video files of each class are imaged
    for class_list_i in class_list:
        class_path = os.path.join (root_path, class_list_i) #path to class directory

        # Get images in each class directory
        for file_name in os.listdir (class_path):
            name, ext = os.path.splitext (file_name) #filename and extension

            # Ignore the original video (.mp4)
            if ext =='.mp4':
                continue
            
            # Add the path of the directory where the video is divided into images and saved
            video_img_directory_path = os.path.join (class_path, name)
            video_list.append (video_img_directory_path)
    
    return video_list

def get_label_id_dictionary (label_dicitionary_path ='./video_download/kinetics_400_label_dicitionary.csv'):
    """
    A function that returns a dictionary that converts Kinetics-400 label names to IDs and a dictionary that converts IDs to label names.
    
    Inputs
    ----------
    label_dictionary_path: str
        Kinetics-400 path to csv file for class label information
    
    Returns
    ----------
    label_id_dict: dict ()
        A dictionary that converts label names to IDs
    id_label_dict: dict ()
        A dictionary that converts IDs to label names
    """

    label_id_dict = dict ()
    id_label_dict = dict ()

    with open (label_dicitionary_path, encoding ='utf-8_sig') as f:

        # Read
        reader = csv.DictReader (f, delimiter =',', quotechar ='"')

        # Read line by line and add to dictionary variable
        for row in reader:
            label_id_dict.setdefault (
                row ['class_label'], int (row ['label_id']) ―― 1)
            id_label_dict.setdefault (
                int (row ['label_id']) --1, row ['class_label'])

    return label_id_dict, id_label_dict

class GroupResize ():
    """
    A class that resizes images together.
    The shorter side of the image is converted to resize (the aspect ratio remains the same).
    """

    def __init __ (self, resize, interpolation = Image.BILINEAR):
        self.rescaler = transforms.Resize (resize, interpolation)
    
    def __call__ (self, img_group):
        return [self.rescaler (img) for img in img_group]

class GroupCenterCrop ():
    """
    A class that clips images together.
    Cut out the image of (crop_size, crop_size).
    """

    def __init __ (self, crop_size):
        self.center_crop = transforms.CenterCrop (crop_size)

    def __call__ (self, img_group):
        return [self.center_crop (img) for img in img_group]

class GroupToTensor ():
    """A class that converts a group of images to torch.tensor."""

    def __init __ (self):
        self.to_tensor = transforms.ToTensor ()
    
    def __call__ (self, img_group):
        # Handle with [0, 255] to match the format of the trained data
        return [self.to_tensor (img) * 255 for img in img_group]

class GroupImgNormalize ():
    """A class that standardizes images together."""

    def __init __ (self, mean, std):
        self.normalize = transforms.Normalize (mean, std)

    def __call__ (self, img_group):
        return [self.normalize (img) for img in img_group]

class Stack ():
    """
    A class that combines image groups into one tensor.
    
    Inputs
    ----------
    img_group: list (torch.tensor)
        List with torch.Size ([3, 224, 224]) as an element
    """

    def __call__ (self, img_group):
        # Since the original training data is BGR, convert the color channel to RGB-> BGR with x.flip (dims = [0])
        Add a dimension for frames with # unsqueeze (dim = 0) and combine with frames dimension
        ret = torch.cat ([(x.flip (dims = [0])). unsqueeze (dim = 0))
                         for x in img_group], dim = 0)
        return ret
       
class VideoTransform ():
    """
    Pre-processing class for files with moving images as images. It behaves differently during learning and inference.
    Since the moving image is divided into images, the divided image groups are collectively preprocessed.
    """

    def __init __ (self, resize, crop_size, mean, std):
        self.data_transform = {
            'train': transforms.Compose ([[
                #DataAugumentation () #None this time
                GroupResize (int (resize)), #Resize images together
                GroupCenterCrop (crop_size), #Crop images together
                GroupToTensor (), # torch.tensor
                GroupImgNormalize (mean, std), #Standardize data
                Stack () # frames Combine in dimension
            ]),
            'val': transforms.Compose ([[
                GroupResize (int (resize)), #Resize images together
                GroupCenterCrop (crop_size), #Crop images together
                GroupToTensor (), # torch.tensor
                GroupImgNormalize (mean, std), #Standardize data
                Stack () # frames Combine in dimension
            ])
        }
    
    def __call__ (self, img_group, phase):
        """
        Parameters
        ----------
        phase:'train' or'val'
            Preprocessing mode specification flag
        """
        return self.data_transform [phase] (img_group)
             
class VideoDataset (data.Dataset):
    """
    Video Dataset.
    Attributes
    ----------
    video_list: list (str)
        List of directory paths that converted videos into images
    label_id_dict: dict ()
        A dictionary that converts class label names to IDs
    num_segments: int
        How many divisions to use the video
    phase:'train' or'val'
        Flag to manage the mode of (training or inference)
    transform: object
        Preprocessing class
    img_template: str
        File name template of the image group you want to load
    """

    def __init__ (self, video_list, label_id_dict, num_segments,
                 phase, transform, img_template ='image_ {: 05d} .jpg'):
        self.video_list = video_list
        self.label_id_dict = label_id_dict
        self.num_segments = num_segments
        self.phase = phase
        self.transform = transform
        self.img_template = img_template

    def __len __ (self):
        '''Return the number of videos'''
        return len (self.video_list)

    def __getitem__ (self, index):
        '''Return the preprocessed image group data, label, label ID, and path'''
        imgs_transformed, label, label_id, dir_path = self.pull_item (index)
        return imgs_transformed, label, label_id, dir_path

    def pull_item (self, index):
        '''Return the preprocessed image group data, label, label ID, and path'''

        #Load images
        dir_path = self.video_list [index]
        indices = self._get_indices (dir_path)
        img_group = self._load_imgs (dir_path, self.img_template, indices)

        # Convert label to ID
        # label = (dir_path.split ('/') [3] .split ('/') [0]) # for Ubuntu
        label = dir_path.split ('/') [3] .split ('\\') [0] # for Windows
        label_id = self.label_id_dict [label]

        # Perform preprocessing
        imgs_transformed = self.transform (img_group, phase = self.phase)

        return imgs_transformed, label, label_id, dir_path

    def _load_imgs (self, dir_path, img_template, indices):
        '''A function that reads and lists images in a batch. '''

        img_group = list ()
        for idx in indices:
            #Load image and add to list
            file_path = os.path.join (dir_path, img_template.format (idx))
            img = Image.open (file_path) .convert ('RGB')
            img_group.append (img)
        
        return img_group
             
    def _get_indices (self, dir_path):
        '''Returns a list of video idx to get when splitting the entire video into self.num_segments'''

        # Number of video frames
        file_list = os.listdir (path = dir_path)
        num_frames = len (file_list)

        #Interval width to acquire video
        tick = (num_frames) / float (self.num_segments)

        # Spacing width of video: List of idx when fetching with tick
        indices = np.array ([int (tick / 2.0 + tick * x)
                            for x in range (self.num_segments)]) + 1
        
        # Example: When extracting 16 frames with 250 frames,
        # tick = 250/16 = 15.625
        #index = [8 24 40 55 71 86 102 118 133 149 165 180 196 211 227 243]

        return indices

if __name__ =='__main__':
    
    #Create data path
    root_path ='./data/kinetics_videos/'
    video_list = make_datapath_list (root_path)
    # print (video_list [0])
    # print (video_list [1])

    #Get video class label and ID
    label_dicitionary_path ='./video_download/kinetics_400_label_dicitionary.csv'
    label_id_dict, id_label_dict = get_label_id_dictionary (label_dicitionary_path)
    # print (label_id_dict)

    # Pre-processing settings
    resize, crop_size = 224, 224
    mean, std = [104, 117, 123], [1, 1, 1]
    video_transform = VideoTransform (resize, crop_size, mean, std)

    #Create Dataset
    val_dataset = VideoDataset (video_list, label_id_dict, num_segments = 16,
                               phase ='val', transform = video_transform,
                               img_template ='image_ {: 05d} .jpg')

    # Data retrieval test
    index = 0
    print (val_dataset.__getitem__ (index) [0] .shape) # image group tensor
    print (val_dataset.__getitem__ (index) [1]) # Label name
    print (val_dataset.__getitem__ (index) [2]) # Label ID
    print (val_dataset.__getitem__ (index) [3]) #path to video
    print ()

    Test with #DataLoader
    batch_size = 8
    val_dataloader = data.DataLoader (val_dataset,
                                     batch_size = batch_size,
                                     shuffle = False)
    
    batch_iterator = iter (val_dataloader) # Convert to iterator
    imgs_transformeds, labels, label_ids, dir_path = next (batch_iterator)
    print (imgs_transformeds.shape)
