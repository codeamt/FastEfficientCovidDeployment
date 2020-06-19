#import packages
import streamlit as st

from fastai import *
import fastai
from fastai.vision import *
from fastai.vision import open_image, load_learner, image, torch
from fastai.callbacks import *

import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO


class ImageEmbeddings():
  '''
  PyTorch Hook for storing linear features of images.
  Args:
    - m --> type:nn.Module --> Linear Layer of model.
  '''
  features=None
  def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None

  def hook_fn(self, module, input, output):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

  def remove(self):
        self.hook.remove()



class TraceMallocMultiColMetric(LearnerCallback):
  """
  Fastai Learner Callback to measures peak RAM usage during each epoch.
  Callbacks and Custom Metrics, Docs: #https://docs.fast.ai/metrics.html#Creating-your-own-metric
  """
  _order=-20 # Needs to run before the recorder
  def __init__(self, learn):
    super().__init__(learn)
    self.train_max = 0

  def on_train_begin(self, **kwargs):
    self.learn.recorder.add_metric_names(['used', 'max_used', 'peak'])

  def on_batch_end(self, train, **kwargs):
    # track max memory usage during the train phase
    if train:
      current, peak =  tracemalloc.get_traced_memory()
      self.train_max = max(self.train_max, current)

  def on_epoch_begin(self, **kwargs):
      tracemalloc.start()

  def on_epoch_end(self, last_metrics, **kwargs):
      current, peak =  tracemalloc.get_traced_memory()
      tracemalloc.stop()
      return add_metrics(last_metrics, [current, self.train_max, peak])


def pre_screen(img, display_img):
  """
  Performs inference and displays target image with result.
  Args:
    - img -> type:Image -> fastai wrapper for pixel image.
    - display_img -> type:str -> location of static image being analyzed.
  """
  with st.spinner('Wait for it...'):
    time.sleep(3)

  model = load_learner('models/', file="e-covidnet.pkl")

  if torch.cuda.is_available():
    model.model.load_state_dict(
        torch.load('models/e-covidnet.pth'))
  else:
    model.model.load_state_dict(
        torch.load('models/e-covidnet.pth',
          map_location=torch.device("cpu")))

  pred_class = model.predict(img)[0]
  pred_prob = round(torch.max(model.predict(img)[2]).item()*100)

  if str(pred_class) == 'COVID-19':
    st.success("COVID-19 with the probability of " + str(pred_prob) + '%.')
    st.image(display_img, width=300)
  elif str(pred_class) == 'pneumonia':
    result = st.success("Viral Pneumonia with the probability of " + str(pred_prob) + '%.')
    st.image(display_img, width=300)
  elif str(pred_class) == 'normal':
    st.success("Normal with the probability of " + str(pred_prob) + '%.')
    st.image(display_img, width=300)



