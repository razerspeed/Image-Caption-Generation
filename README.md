# Image Caption Generation
In this project I have implemented the encoder decoder network for image caption generation . 
The CNN model act as Encoder and the Transformer Decoder act as an Decoder
![alt text](cnn_transformer.jpg)

This model is deployed in Heroku .

The encoder is deployed as a REST service in SERVER 1  https://image-caption-server1.herokuapp.com

The decoder is deployed as a REST service in SERVER 2 https://image-caption-sever2.herokuapp.com

![alt text](diagram.jpg)

The website is made using Streamlit Framework 


link to the final web app- https://image-caption-website.herokuapp.com/

NOTE : First time the website me LOAD SLOW as the Heroku server sleep after some time of inactivity.

If anyone wants to replicate the project

clone these repo seperately and make heroku app out of it

https://github.com/rohit-testing-ml/image-caption-server1

https://github.com/rohit-testing-ml/image-caption-server2

https://github.com/rohit-testing-ml/image-caption-website
