#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf

filepath = "model/Rice_Leaf_Disease_Detection xception Model.h5"
model=load_model(
    filepath, custom_objects=None, compile=True,
)

print('@@ Model loaded')

def pred_dieas(leaf):
  test_image = load_img(leaf, target_size = (224,224)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result) # get the index of max value
  pred=3
  if pred == 0:
    return "BrownSpot Diseased Leaf", 'BrownSpot.html' 
  elif pred == 1:
      return 'Healthy Leaf', 'Healthy.html' 
  elif pred == 2:
      return 'LeafBlast Diseased Leaf', 'LeafBlast.html'  
  else:
    return "Healthy Leaf", 'Healthy.html' 
    

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_dieas(leaf=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,) 
    
    