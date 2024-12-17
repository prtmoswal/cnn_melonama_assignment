# Upgrad CNN Melanoma Detection Assignment

#### Problem Statement
In this assignment, you will build a multiclass classification model using a custom convolutional neural network in tensorflow. <br />

Problem statement: To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant. The data set contains the following diseases: <br />

Actinic keratosis <br />
Basal cell carcinoma <br />
Dermatofibroma <br />
Melanoma <br />
Nevus <br />
Pigmented benign keratosis <br />
Seborrheic keratosis <br />
Squamous cell carcinoma <br />
Vascular lesion <br />
NOTE: You don't have to use any pre-trained model using Transfer learning. All the model building process should be based on a custom model. <br />

Project Pipeline <br />
Data Reading/Data Understanding → Defining the path for train and test images <br />
Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180. <br />
Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset <br />

#### Model Building & training : Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training <br />
Train the model for ~20 epochs <br />
Write your findings after the model fit, see if there is evidence of model overfit or underfit <br />

#### Choose an appropriate data augmentation strategy to resolve underfitting/overfitting Model Building & training on the augmented data :
Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1). <br />
Choose an appropriate optimiser and loss function for model training <br />
Train the model for ~20 epochs <br />
Write your findings after the model fit, see if the earlier issue is resolved or not? **Class distribution: ** <br />
Examine the current class distribution in the training dataset <br />
Which class has the least number of samples? <br />

#### Which classes dominate the data in terms of the proportionate number of samples? Handling class imbalances:
Rectify class imbalances present in the training dataset with Augmentor library. Model Building & training on the rectified class imbalance data: <br />
Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1). <br />
Choose an appropriate optimiser and loss function for model training <br />
Train the model for ~30 epochs <br />
Write your findings after the model fit, see if the issues are resolved or not? <br />
The model training may take time to train and hence you can use Google colab. <br />
