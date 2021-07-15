import torch
import torch.utils.data as data

from networks import ECO_Lite
from  dataloader import (
    make_datapath_list, get_label_id_dictionary,
    VideoTransform, VideoDataset
)
from prepare import load_pretrained_ECO

def show_eco_inference_result(dir_path, outputs_input, id_label_dict, idx=0):
    """ Function that outputs the top 5 inference results for each data in the mini-batch """
    print('File：', dir_path[idx])  # printing file name

    outputs = outputs_input.clone()     # making a copy

    # Display from 1st to 5th place
    for i in range(5):
        output = outputs[idx]
        _, pred = torch.max(output, dim=0)  # Predict the label of the maximum probability
        class_idx = int(pred.numpy())       # Output class ID
        print('Prediction # {}：{}'.format(i + 1, id_label_dict[class_idx]))
        outputs[idx][class_idx] = -1000     # making the value smaller

def main():
    # Image group path creation
    root_path = './data/kinetics_videos/'
    video_list = make_datapath_list(root_path)

    # Create a dictionary of video class labels and IDs
    label_dicitionary_path = './video_download/kinetics_400_label_dicitionary.csv'
    label_id_dict, id_label_dict = get_label_id_dictionary(label_dicitionary_path)

    # Pre-processing settings
    resize, crop_size = 224, 224
    mean, std = [104, 117, 123], [1, 1, 1]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    # Creating a Dataset
    val_dataset = VideoDataset(video_list, label_id_dict, num_segments=16,
                               phase='val', transform=video_transform,
                               img_template='image_{:05d}.jpg')

    # DataLoader to load the data
    batch_size = 8
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
    batch_iterator = iter(val_dataloader)  # Convert to iterator
    imgs_transformeds, labels, label_ids, dir_path = next(batch_iterator)

    # Instantiate the model 
    net = ECO_Lite()
    # Change to inference mode
    net.eval()
    # Load the trained model
    net.load_state_dict(torch.load('./models/pretrained.pth'))

    # Inference with ECO
    with torch.set_grad_enabled(False):
        outputs = net(imgs_transformeds)

    # Make predictions
    idx = 0
    show_eco_inference_result(dir_path, outputs, id_label_dict, idx)

if __name__ == '__main__':
    main()