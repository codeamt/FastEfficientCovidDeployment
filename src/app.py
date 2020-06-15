#import packages
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
from fastai import *
from fastai.vision  import *
from fastai.vision import open_image
import PIL.Image
import requests
from io import BytesIO
from helpers import pre_screen, TraceMallocMultiColMetric

## Frontend Design

#STYLES
with open("style.css") as f:
  st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#SIDEBAR
st.sidebar.title("Fast Efficient CovidNet")

st.sidebar.markdown("""
  An inference API for pre-screening upper-respitory infectious diseases based on Chest XRays (CXR) images.
  """, unsafe_allow_html=True,)


st.sidebar.info(
        " [View source code on GitHub](https://github.com/codeamt/udacity-mle-capstone-project)."
    )
st.sidebar.header("About")

st.sidebar.markdown("""
  This model builds on research first presented by authors of [Covid-Net](https://arxiv.org/pdf/2003.09871.pdf). Using a pre-trained EfficientNet architecture and a fasta.ai classifier, the project produces state-of-the-art results, trained with fewer FLOPs on the [COVIDx](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md) dataset discussed on [GitHub](https://github.com/lindawangg/COVID-Net).<br><br>
  **Maintained by:** <br>AnnMargaret Tutu<br>
  [codeamt.github.io](https://codeamt.github.io)
  """, unsafe_allow_html=True,)



#MAIN CONTENT
option = st.radio('', ['Choose a test image', 'Choose your own image'])
if option == 'Choose a test image':
  test_images = os.listdir('models/data/test/')
  test_image = st.selectbox(
        'Please select a test image:', test_images)
  # Read the image
  file_path = 'models/data/test/' + test_image
  img = open_image(file_path)

  # Get the image to display
  display_img = mpimg.imread(file_path)

  # Predict and display the image
  pre_screen(img, display_img)

else:
  url = st.text_input("Please input a url:")
  if url != "":
    try:
      # Read image from the url
      response = requests.get(url)
      pil_img = PIL.Image.open(BytesIO(response.content))
      pil_img.resize((240,240))
      display_img = np.asarray(pil_img)
      img = image.pil2tensor(img, np.float32).div_(255)
      img = image.Image(img)

      # Predict and display the image
      pre_screen(img, display_img)
    except:
      st.text("Invalid url!")

