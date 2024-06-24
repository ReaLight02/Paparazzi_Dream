# Introduction
This repository was created for the purpose of allowing the team members to work together in a simple way. The main objective is to create a CNN model that trains itself on a dataset and tries to predict the outcomes on images it has never seen before. The secondary objective is to apply filters on these images.

## Repository organization
&nbsp;&nbsp;&nbsp;&nbsp;CNN <br>   	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CNN.py <br> 	*CNN model and its training*
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cropping.py <br> *function needed to crop the images using haar cascade*
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_loader.py <br> *data augmentation and creation of the training/validation data*
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mapping.py <br> *creation of classes and mapping of face to ID*
&nbsp;&nbsp;&nbsp;&nbsp;Dataset <br> *original dataset*
&nbsp;&nbsp;&nbsp;&nbsp;Filters <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;filters.py <br> *functions needed in order to apply filters to an image*
&nbsp;&nbsp;&nbsp;&nbsp;Prediction <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;InPrediction.py <br> *functions needed in order to use a previously trained model to predict an input image*
&nbsp;&nbsp;&nbsp;&nbsp;Utilites <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;haarcascade_eye.xml <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;haarcascade_frontalface_default.xml <br>
&nbsp;&nbsp;&nbsp;&nbsp;README.md <br>

## Team members
[Mocanu Alecsandru](https://github.com/ReaLight02) <br>
[Ene Alexadru](https://github.com/ScherzoNo) <br>
[Frau Antonio](https://github.com/ShinobuSmile) <br>

