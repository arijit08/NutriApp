import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, render_template
import numpy as np
from PIL import Image

app = Flask(__name__)

class_names = ["chicken_curry","chicken_wings","fried_rice","grilled_salmon","hamburger","ice_cream","pizza","ramen","steak","sushi"]
model = tf.keras.models.load_model("NutriApp/nutriapp.h5", custom_objects={'KerasLayer':hub.KerasLayer})

def read_image(image_path):
  image =  tf.io.read_file(image_path) # reading image file
  # name, exten = os.path.splitext(image_path)
  image = tf.image.decode_image(image,channels=3)
  image =  tf.image.resize(image, size = [224,224])# resize image
  image = image/255 # normalise
  return image

def predict_img(model, img, pred_type, class_names):
  #image = read_image(filepath)
  img_np = tf.keras.utils.img_to_array(img)
  img_tf = tf.convert_to_tensor(img_np, dtype=tf.float32)

  #img_tf = tf.image.decode_image(img_tf,channels=3) #decode
  img_tf =  tf.image.resize(img_tf, size = [224,224])# resize image
  img_tf = img_tf/255 # normalise

  prediction =  model.predict(tf.expand_dims(img_tf,axis = 0))

  if pred_type=="binary":
    image_category = str(np.where(prediction > 0.5,class_names[1],class_names[0]))
  else:
    image_category = str(class_names[np.argmax(prediction)])
  return image_category

@app.route('/')
def loadpage():
  return render_template("index.htm")

@app.route('/classify', methods = ['POST'])
def classify():
  data = request.files.get("image")
  if data:
    img = Image.open(data.stream)
  else:
    return "Error, no 'image' found"
  print("img size = " + str(img.size))
  label = predict_img(model,img, "multiclass", class_names)
  if label:
      return label
  else:
      return "Could not classify"

if __name__ == "__main__":
  app.run(host="0.0.0.0",port=8080)
  #app.run(debug=False)