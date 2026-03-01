# Building an Image Classifier with a Convolutional Neural Network (CNN)

*Full name:* Yvar Joseph

*Project Description*: 
This project is designed to construct a Machine Learning model which implements a Convolutional Neural Network (CNN)
to classify various numbers between zero and nine. Furthermore, the project implements various ML libraries to achieve the desired 
goal i.e. PyTorch, Sklearn and Matploblib (for plotting purposes), to name a few. To gain a deeper understanding, it is recommended
that the reader follow the description of each class and function, to obtain a holistic view of the model's operation and overall performance.

## Summary
   In the world of Artificial Intelligence (AI), multiple state-of-the-art models are developed on a daily basis -- some of the most impressive
   of which are used for vision applications e.g. EV automation, disease classification and more. However, despite most of the recent models being
   of the transformer type, one model which is still used today, and ever-relevant for the comprehension of the more advanced models, is the Convolutional
   Neural Network (CNN). 

   This project thus seeks to develop a CNN image model to classify numbers between zero and nine - an important application for vision models. 
   The initial steps begin by importing the various libraries which are relevant to achieving such a goal, namely PyTorch, Sklearn, Numpy (for 
   common mathematical operations) and Matploblib for graphical operations. 

   Furthermore, upon importing all the necessary libraries, a custom dataset is built to organize the imported dataset from PyTorch's vision
   dataset library. This ensures the data is normalized appropiately for model training and classification. Henceforth, the model is built from
   scratch by implementing OOP principles, developing a model class inheriting from PyTorch's nn.Module class, which provides certain features 
   to our model, such as the overridden *forward()* method, which makes passing data to our model straightforward.

   Finally, once the model is built, a training loop function is established which is designed to keep track of the model's losses i.e. 
   how well it predicts unforseen data not found in the training dataset. Depending on the results, the backpropagation algorithm is applied along
   with an optimizer (to perform gradient descent) to update the model's weights to improve it's ability to make accurate predictions. In the end, 
   the model obtains an accuracy of 97% on training data and 100% on validation (new) data. 

   For a more comprehensive understanding of the model and its implementation, it is recommended that the full process be read. Each
   cell block seen in the notebook above is described in plenary below. 😄

## Extra Project Details

## Importing the Relevant Libraries

As with any Machine Learning project, the necessary libraries are first imported to a smooth development process is achieved.
Thus, certain libraries are imported, some of the most relevant libraries are described in plenary: 
 
 - torch: the PyTorch library
 - Dataset & DataLoader: PyTorch library designed to create, organize and split a dataset into various batches
 - torch.optim: optimizer used during backpropagation, ensuring the model learns efficiently
 - sklearn.metrics: Sklearn's metrics classes and methods to be used for measuring model performance

Setting the device - upon importing all necessary libraries, the device is set to either CPU or GPU (if available). 
```python
#Set the system device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("System device: {}".format(device))
```

## Loading the Dataset 
The next step involves loading the dataset: the MNIST dataset, which is directly forked from the PyTorch datasets.vision library.
The fastest way to import this dataset is displayed in cell block three, documented below for easy reference: 

```python
#Loading and preprocessing the dataset
mnist_root_dir = "main/mnist"
mnist_dataset_train = MNIST(mnist_root_dir, train = True, download = True)
mnist_dataset_val = MNIST(mnist_root_dir, train = False, download = True)
```
The code block above gathers the training and validation splits of the MNIST dataset by setting train to *True* and *False* respectively.
This data split is essential, as it aids us in identifying the model's weakpoints i.e. if it's overfitting or underfitting. Henceforth, the
next few steps are to be followed: 
  1. Divide the dataset into training and validation sets (which has already been done for us in the previous step)
     ```python
      #Dividing the dataset into training and validation sets
      train_images = mnist_dataset_train.data
      train_targets = mnist_dataset_train.targets
      val_images = mnist_dataset_val.data
      val_targets = mnist_dataset_val.targets   
     ```

  2. Place the classes (labels) into their own list
     ```python
       #Displaying all the classes found in the dataset
       target_classes = mnist_dataset_train.classes
     ```
## Creating the CNN Model Dataset Class 
```python
#Developing the custom dataset for MNIST called MNISTDataset
class MNISTDataset(Dataset):
     def __init__(self, x, y):
         #ensure both x and y can are of type Tensor
         if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)): #1
            x = torch.Tensor(x)
            y = torch.Tensor(y)

         x = x.float() / 255 # 2
         x = x.view(-1, 1, 28, 28) #3
         self.x, self.y = x, y

     def __getitem__(self, idx):
         return self.x[idx].to(device), self.y[idx].to(device) #4

     def __len__(self):
         if len(self.x) == 0:
            return 0

         return len(self.x)

     def __str__(self):
         #String summary of the dataset data and targets
         return "Dataset length: {}\n Dataset target length: {}" \
                .format(len(self.x), len(self.y))
```
Line 1 first ensures that both inputs are of the same type, in this case Tensors. Hereafter, line 2 
nornalizes the data (image matrix) by dividing it by 255. Then, 3 reshapes the data into a 4-dimensional
dataset, which is required for the CNN model architecture. Line 4 defines the __getitem__ dunder method, which
returns a single (input, label/class) output from the dataset.

## The CNN Model Class
```python 

#Developing the Convolutional Neural Network Model
class CustomResNetModel(nn.Module): #Similar to ResNet18 but custom
     def __init__(self):
         super().__init__()
         self.layer1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2)
         self.maxPool = nn.MaxPool2d(2, stride = 2)
         self.layer2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
         self.layer3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
         self.layer4 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
         self.layer5 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
         self.layer6 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
         self.layer7 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
         self.avgPool = nn.AvgPool2d(2, stride = 2)
         self.flatten = nn.Flatten()
         self.out = nn.Linear(2048, 10)
         #TO DO - Add BatchNorm if required

     def forward(self, x):
         x = self.layer1(x)
         x = self.maxPool(x)
         x = self.layer2(x)
         x = self.layer3(x)
         x = self.layer4(x)
         x = self.layer5(x)
         x = self.layer6(x)
         x = self.layer7(x)
         x = self.flatten(self.avgPool(x))
         return self.out(x)
```
The code above displays the creation of a Convolutional Neural Network with its respective architectural design. In the *forward()* 
method the architecture is manually constructed in the correct order, ensuring the model can operate in a robust manner. 
