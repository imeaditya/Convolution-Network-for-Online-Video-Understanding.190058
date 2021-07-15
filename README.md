## BCS_ECO-Teams1-
### Efficient Convolutional Network for Online Video Understanding
 
#### network.py
We have created the network of eco-lite in network.py file consisiting of 2 parts ECO-2D and ECO-3D.
where ECO-2D consist of basic convolution layers and different inception layers while our ECO-3D consist of resnet3D and gloabal average pooling layer.
Finally we place ECO-3D on top of ECO-2D and fully connected layer on top of ECO3D.

#### prepare.py
We have downloaded the pre-trained wieghts of ECO-model trained in kinetics-400 dataset from official repo and loaded the weights into our model, As training our model from scratch on dataset like kinetics400 is impossible on colab.

#### Dataset downloading and processing
We have downloaded the kinetics dataset it has size of around 16.5gb not possible to store in drive so we just make a small subset of that large csv file with few classes and downloaded the dataset using download.py file and then extract 30 frames from each mp4 file using ffmpeg library for further use.

#### Dataloader.py
We have divided this task into 4 subparts. First one is creating a list of data paths to the directory where image data of videos is stored. Then defining function to return a dictionary that converts dataset label names to IDs and a dictionary that converts IDs to label names. Now since the files with moving images as images behaves differently during learning and inference, they are divided into images third task is to collectively preprocess theses divided image groups. And then the fourth task is to create datset returning the preprocessed image group data, label, label ID, and path.

#### Inference.py 
We loaded the dataset into our model with pre-trained wieghts and infer or evaluate our model with some kinetics videos. We loaded the UCF101 dataset as this is very large datset even its extracting take so much time we have to remove kinetics dataset from our drive.

#### Further Goals
<ol>
<li> we will try to evaluate our model in ucf101 dataset and then after getting accuracy on our smaller subset dataset we will try to finetune the model and also try to train model with only wieght updation allowed in final layer. 
<li> try to recognize action in online video(live video) where we have to queue up the input from previous frames and merges it with new video frames comming 
</ol>

### Contribution 

**Ankit** : Wrote doc detail of (intro and conclusion)literature review in PPT, wrote the literature review of ECO paper,network achitecture of Conv2d, Conv3D layer and Resnet3D in network architecture and Eco-lite network architechture in pytorch. <br/>
**Lochan** : Contributed in Paper review and Doc, Slides in PPT, preprocessed the kinetic dataset with loading it and its pre-trained weights into the model and evaluate the result though inference.py file. <br/>
**Utkarsh** : Wrote Paper review, doc and slides preparation for mid eval, speaker in mid eval presentation, Added inception layer to ECO2D, Loading pre-trained weights of kinetic dataset into the model and evaluated on the network model, Extracted UCF dataset into frames of each class.<br/>
**Vinamra** : Wrote Paper review, implemented Eco-lite network architecture in pytorch, wrote dataloader scripts for test and validation dataset, convert kinetics video data to image data(frames) using ffmpeg and also try to finetune model with ucf101 dataset.
   


   
