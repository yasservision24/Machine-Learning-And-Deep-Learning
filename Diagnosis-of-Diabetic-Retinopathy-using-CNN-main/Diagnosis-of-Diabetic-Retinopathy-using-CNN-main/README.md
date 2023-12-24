# Diagnosis-of-Diabetic-Retinopathy-using-CNN

## Overview
This model has been trained using a Convolutional Neural Network to classify images as either having Diabetic Retinopathy or not.

The prevalence of Diabetic Retinopathy is alarmingly high, affecting a significant proportion of individuals with long-standing diabetes. Early detection and timely treatment are crucial for preventing vision loss and improving patient outcomes. However, manual interpretation of retinal images for Diabetic Retinopathy screening can be time-consuming and subject to human error. Therefore, there is a pressing need for an automated and accurate tool that can assist healthcare professionals in grading the severity of Diabetic Retinopathy.
The existing methods for detecting and grading Diabetic Retinopathy often rely on subjective assessments and extensive manual labor, leading to inefficiencies and potential inconsistencies in diagnosis. Moreover, the increasing prevalence of diabetes and the limited availability of ophthalmologists further exacerbate the challenges in timely screening and diagnosis. Therefore, there is a need to develop a robust and reliable automated system that can accurately detect and grade Diabetic Retinopathy, enabling early intervention and personalized treatment plans.


Data Description :
This dataset consists of a large collection of high-resolution retinal images captured under various imaging conditions. A medical professional has assessed the presence of Diabetic Retinopathy in each image and assigned a rating on a scale ranging between 0 and 1, which corresponds to the following categories:

Diabetic Retinopathy ---> 0
No Diabetic Retinopathy ---> 1

## Dataset
The dataset used for training the model can be downloaded from Kaggle: [Diabetic Retinopathy Dtaset](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy).
It contains three subdirectories: 'train',valid and 'test,' each containing images of positive and negative cases.
Or you can use API keykaggle datasets download -d pkdarabi/diagnosis-of-diabetic-retinopathy

## Using the model
You will have to generate your Kaggle token to upload data to collab from kaggle Or you can download it on you local machine
