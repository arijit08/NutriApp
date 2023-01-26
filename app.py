import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, render_template
import numpy as np
from PIL import Image

#Create Flask application - to create a server instance and pass requests to this app
app = Flask(__name__)

#Recognised food items
class_names = ["chicken_curry","chicken_wings","fried_rice","grilled_salmon","hamburger","ice_cream","pizza","ramen","steak","sushi"]
#Pre trained model
model = tf.keras.models.load_model("NutriApp/nutriapp.h5", custom_objects={'KerasLayer':hub.KerasLayer})

#Function that passes image to the model and returns its output
def predict_img(model, img, pred_type, class_names):
  #Convert PIL image to np array
  img_np = tf.keras.utils.img_to_array(img)
  
  img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)

  #Resize and normalise image
  img_tf =  tf.image.resize(img_tf, size = [224,224])# resize image
  img_tf = img_tf/255 # normalise

  #Actual prediction
  prediction =  model.predict(tf.expand_dims(img_tf,axis = 0))

  if pred_type=="binary":
    image_category = str(np.where(prediction > 0.5,class_names[1],class_names[0]))
  else:
    #This code is multiclass (10 possible food items). Find prediction label with highest probability
    image_category = str(class_names[np.argmax(prediction)])
  return image_category

@app.route('/')
def loadpage():
  #In case of get request on the root address, return the following html file (stored in templates folder)
  return render_template("index.htm")

#Following code handles POST request to root_address/classify
@app.route('/classify', methods = ['POST'])
def classify():
  #get image from HTTP POST request
  data = request.files.get("image")
  if data:
    img = Image.open(data.stream)
  else:
    return "Error, no 'image' found"
  #Pass PIL image to get final classification
  label = predict_img(model,img, "multiclass", class_names)
  if label:
      return label
  else:
      return "Could not classify"

#In deployment, run on 0.0.0.0 host and 8080 port, in debugging/local run the second line instead
#NOTE: run on debug=False, else cd to the folder where app.py is stored before running. There seems to be a bug here
if __name__ == "__main__":
  app.run(host="0.0.0.0",port=8080)
  #app.run(debug=False)