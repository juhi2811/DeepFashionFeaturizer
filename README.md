# DeepFashionFeatures
This is a cloth recognizer multi-output deep-learning classification model based on Functional API, Keras and tensorflow. The model is trained on the famous open source DeepFashion dataset. It takes as input the images of the deep fashion dataset and categorizes clothes by three attributes:
- purpose of clothing
- design of clothing
- occassion

The network architechture of the multi-output model is as shown:

![image](https://user-images.githubusercontent.com/51826271/184906016-e7a142bc-a559-48a4-ad03-af85937eb0fe.png)

Based on the output of the multi-output, we then predict the print type of the model. The model architecture of the second model is as shown below:
