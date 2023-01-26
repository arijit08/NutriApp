# NutriApp
 Recognise photo of food items and display nutrition information of the same

This application uses Flask to develop a web app/server

Get request to the server returns a basic webpage which allows the user to select an image from their local machine

Doing so sends a Post request with the image, and allows app.py to classify the image based on the saved deep learning model

Deep learning model itself was trained via transfer learning from imagenet model on tensorflow website, along with few more images from a training dataset

Currently the model can only recognise 10 classes:
"chicken_curry","chicken_wings","fried_rice","grilled_salmon","hamburger","ice_cream","pizza","ramen","steak","sushi"