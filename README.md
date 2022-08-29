# DeepFashionFeatures
This is a cloth recognizer multi-output deep-learning classification model series based on Functional API, Keras and tensorflow. The model series that first categorizes clothes by basic attributes. The second model takes as input the image and the output of the first model. The model is trained on the famous open source DeepFashion dataset. It takes as input the images of the deep fashion dataset and categorizes clothes by three attributes:
- purpose of clothing
- design of clothing
- occassion

![network_series](https://user-images.githubusercontent.com/51826271/187241107-ed58b694-e8d1-40c0-918d-414c1e8c7424.png)

The network architechture of the multi-output model is as shown:

![image](https://user-images.githubusercontent.com/51826271/184906016-e7a142bc-a559-48a4-ad03-af85937eb0fe.png)

Based on the output of the multi-output, we then predict the print type of the model. The model architecture of the second model is as shown below:

![image](https://user-images.githubusercontent.com/51826271/185932856-2b91e1ff-54e8-4420-a4b4-c51f19a5ed41.png)
