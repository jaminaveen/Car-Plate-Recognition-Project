# Car-Plate-Recognition-Project
End-to-End pipeline for car number plate recognition using Deep Learning

### Contents
 1.  YoloV3 for License plate detection
 2.  Segmentation of digits using Image Pre-processing techniques
 3.  Trained CNN classifier for classifying segmented digits
 4.  Training pipeline for CRNN Optical Character Recognition model for predicting text on the detected plate (Model needs to be trained on more data to increase efficiency)
 5.  Tutorial notebooks
 6.  Flask application for inference using combination of YOLO followed by segmentation, and CNN classification
 7.  AWS Deeplens tutorial using pre-trained SSD model aws artifiacts
 8.  Airflow training pipelines
           
           - CNN Training pipeline
           - CRNN Training pipeline
 9.  Dockerizing airflow pipelines
            
            To fire up docker service, clone this repository and cd into this folder
            - docker-compose -f docker-compose-CeleryExecutor.yml up
            - docker-compose -f docker-compose-CeleryExecutor.yml up -d (detached mode)
 
#### Object Detection using YOLOv3:
     Prior work on object detection repurposes classifiers to perform detection. Instead, object detection is framed as a regression            problem to spatially separated bounding boxes and associated class probabilities. 
     A single neural network predicts bounding boxes and class  probabilities directly from full images in one evaluation. Since the whole      detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.
 
#### AWS DeepLens
     AWS DeepLens is a deep learning-enabled video camera. It is integrated with the Amazon Machine
     Learning ecosystem and can perform local inference against deployed models provisioned from the AWS Cloud.
     
##### Supported Modeling Frameworks:
      With AWS DeepLens, you train a project model using a supported deep learning modeling framework.
      You can train the model on the AWS Cloud or elsewhere. Currently, AWS DeepLens supports Caffe, TensorFlow and Apache MXNet frameworks
      
##### Flask App - Sample Prediction

![image](https://user-images.githubusercontent.com/37238004/70835944-a65f0380-1dcc-11ea-8def-d4bda672fbf8.png)
      
      
#### Report Link
https://docs.google.com/document/d/1Sa3jxZ_6bPCQz6wDBn-h-yO2jaeonfrUAkS2-ZWV7hU/edit?usp=sharing
      
