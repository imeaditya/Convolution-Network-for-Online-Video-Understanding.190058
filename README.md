## BCS_ECO-Teams1-
### Efficient Convolutional Network for Online Video Understanding
 
#### network.py
I have created the network of eco-lite in network.py file consisiting of 2 parts ECO-2D and ECO-3D.
where ECO-2D consist of basic convolution layers and different inception layers while the ECO-3D consist of resnet3D and gloabal average pooling layer.
Finally I placed ECO-3D on top of ECO-2D and fully connected layer on top of ECO3D.

#### prepare.py
I have downloaded the pre-trained wieghts of ECO-model trained in kinetics-400 dataset from official repo and loaded the weights into our model, As training the model from scratch on dataset like kinetics400 is impossible on colab.

#### Dataset downloading and processing
I have downloaded the kinetics dataset it has size of around 16.5gb not possible to store in drive so I just make a small subset of that large csv file with few classes and downloaded the dataset using download.py file and then extract 30 frames from each mp4 file using ffmpeg library for further use.

#### Dataloader.py
I have divided this task into 4 subparts. First one is creating a list of data paths to the directory where image data of videos is stored. Then defining function to return a dictionary that converts dataset label names to IDs and a dictionary that converts IDs to label names. Now since the files with moving images as images behaves differently during learning and inference, they are divided into images third task is to collectively preprocess theses divided image groups. And then the fourth task is to create datset returning the preprocessed image group data, label, label ID, and path.

#### Inference.py 
I loaded the dataset into our model with pre-trained wieghts and infer or evaluate our model with some kinetics videos. I loaded the UCF101 dataset as this is very large datset even its extracting take so much time we have to remove kinetics dataset from our drive.

#### Further Goals
<ol>
<li> I will try to evaluate our model in ucf101 dataset and then after getting accuracy on our smaller subset dataset I will try to finetune the model and also try to train model with only wieght updation allowed in final layer. 
<li> try to recognize action in online video(live video) where I have to queue up the input from previous frames and merges it with new video frames comming 
</ol>

   


   
