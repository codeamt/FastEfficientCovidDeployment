import os
import base64
import json
import requests
import time

from flask import render_template, redirect, request, session, url_for
from werkzeug.utils import secure_filename
#from multiprocessing import Process

from fastai import *
from fastai.vision import *
import fastai
import torch
import pytorchcv
from app import app
from app.inference import CovidEfficientScan


def run_inference(file):
  """
  Classifies input image file using CovidEfficient algorithm.
  args:
    file --> type:FileStorage
    - flask Filestorage from HTTP POST request.
  """

  #Categorical label map for populating classification result text in HTML view.
  label_map = {
  "normal": "Normal",
  "pneumonia": "Pneumonia",
  "COVID-19": "COVID-19"
  }

  #Instantiate inference model
  eff_covid = CovidEfficientScan(app.config["PATH"])
  print(eff_covid)

  #Store filename and destination folder
  fname = secure_filename(file.filename)
  upload_folder = app.config['UPLOAD_FOLDER']

  #Construct target url and save.
  file_path = os.path.join(upload_folder, fname)
  file.save(file_path)

  #Get byte data for inference
  file_data = pathlib.Path(file_path).read_bytes()
  print("classifying xray...")
  img = open_image(BytesIO(file_data))

  #Perforem timed inference
  t = time.time()
  pred_class, pred_idx, outputs = eff_covid.model.predict(img)
  dt = time.time() - t
  print(outputs, pred_idx)
  print("done. execution time: %0.02f" % (dt))
  print("Image %s classified as %s" % (pathlib.Path(file_path).name, pred_class))
  print("prob", float(outputs[0]))

  #Return labels and probabilities
  return [label_map[str(pred_class)], float(outputs[0])]



@app.route('/')
def index():
  """
  Index Endpoint
  """
  return render_template('index.html')

@app.route('/static/uploads',  methods=['POST'])
def uploads():
  """
  Inference Endpoint
  """
  uploaded_files = request.files.getlist("input-folder-3[]")
  if uploaded_files is None:
    #bootstrap fileinput response needs to be an object, otherwise throws error.
    return {}
  else:
    #iterate file storages
    for item in uploaded_files:
      #pass FileStorage item  for inference result (tuple).
      result = run_inference(item)
      #slice into result for label and stringified probability
      label = result[0]
      prob = str(result[1])
      #construct classification result view for bootstrap fileinput
      updated_img = '<img src="' + file_path + '" "class=file-preview-image" title="Classified as: ' + label +'"style="margin-bottom:5px !important" width="70%" height="70%"><div class="file-footer-caption" title="COVID-19 _13_.png"><div class="file-caption-info" style="color: #fff; font-weight:600">'+fname+'</div><div class="file-size-info"><samp style="color: #fff; font-weight:600">' + label + ', (%) ' + prob +'</samp></div>'
      #return view as value for initial prewiew value.
      return {"initialPreview": updated_img}



