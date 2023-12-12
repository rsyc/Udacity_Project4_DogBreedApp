[//]: # (Image References)

[image1]: ./images/sample_dog_output2.png "Sample Output"


# Dog Breed Classifier

## Project Introduction

This project is about writting a dog identifier algorithm. Given an image of a dog, the algorithm will identify an estimate of the dog’s breed (example is given below). If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]


## Files and folders description

-- **haarcascade folder:** This forlder includes an _.xml_ file named _haarcascade_frontalface_alt.xml_ which is an OpenCV pretrained face detector. We have used use this detector to find human faces.

-- **images folder:** This folder contains some of the sample images used (or can be used) at the end of this project to see what the model predictions are.  

-- **saved_models folder:** This folder contains outputs (weights) of CNN models created/modified in this project.

-- **bottleneck_features folder:** In this folder we have saved pre-computed bottleneck features from different pretrained CNN models available in Keras including VGG-16, VGG-19, ResNet-50, Inceprion, and Xception. In Step 4 and 5, we use transfer learning to create a CNN using these bottleneck features. 

-- **dog_app.ipynb:** This is the main jupyter notebook where all the steps towards building a dog breed classifier algorithm are explained. 

-- **dog_app html file:** The html file of the jupyter notebook including the results.
 
-- **extract_bottleneck_features.py** This python code includes helping functions which help to extract bottleneck features (weights) from different pretrained models which then be used in our models for predictions. 

-- **LICENSE:** A .txt file including license information.

## Results

The notebook includes different steps:

-- Step 0: Import Datasets
	- Here we import both dog and human datsets. Dog dataset includes tarining, validation, and testing datasets. 

-- Step 1: Detect Humans
	- At this step we have used a pre-trained face detector model using OpenCV's implementation of Haar feature-based cascade classifiers. A function is then been created to use this model to define whether the image belongs to a human or not (`face_detector`).

-- Step 2: Detect Dogs
	- Here we used a pre-trained ResNet-50 model to detect dogs in images. Two functions are then created to used this model and determine if the input image belong to a dog or not (`dog_detector`). 

-- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
	- At this step we first try to create our own convolutional neural network (CNN) that can classify dog breeds. A random guess will provide a correct answer roughly 1 in 133 times (133 breeds), which corresponds to an accuracy of less than 1%. Here we have used three CNN layers each followed by a max_pooling layer. The last two layers include one global_average_pooling to flatten the input tensor into a 1D vector and a fully connected layer(Dense) with 133 nodes that corresponds to the 133 different dog breeds. Using this simple CNN model the accuracy of the model applied on test dataset increased to 2.8708%. We can increase the performance of the CNN model by increasing the layers and optimising and feature tuning of the model. We can also increase the number of epochs. However, this takes very long time to train such model as it gets more complicated with more layers. Also we will be needing computers with high GPUs. Therefore, another better method is using already trained CNN models, which we have looked at in more details in steps 4 and 5.

-- Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
	- At this step we used transfer learning using the features from VGG-16 pretrained model. This helps to reduce training time without sacrificing accuracy. The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as an input to our model. Our model includes a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax activation function for multiclass classification. With this model the accuracy on test dataset increased to 41.2679%.

-- Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
	- To increase the accuracy of the model, here we used other pretrained models and tested them on the test dataset to choose the one with the best accuracy for our algorithm. Using features from the Xception pretrained model showed the best performance with the accuracy of about 85% on the test dataset. 

-- Step 6: Write your Algorithm
	- At this step we write the algorithem that uses `dog_detector` and `face_detector` functions to detect whether the image belongs to a human, dog, or neither of them. If the image belongs to a dog then it uses our transfer-learned-model to predict the breed. On the other hand, if human face is detected, then the function tries to find the best match of dog breed to that face. At the end, if the image contains no dog or human, it notifies the user with a message.  

-- Step 7: Test Your Algorithm
	- At this last step we try our algorithm on couple of sample images of 2 human faces, 3 dog images, and one image that is neither a dog nor a human. Here we also suggest three possible points of improvement for the algorithm.


The main findings of this project can also be found at the post available [here](https://medium.com/@rojan.saghian/lets-find-your-dog-breed-3b0eb5edbb3a).




## Licensing

For licensing information please refer to LICENSE.txt


