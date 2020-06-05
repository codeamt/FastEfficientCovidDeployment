from fastai import *
from fastai.vision import *
import fastai
import torch
import pytorchcv


class CovidEfficientScan(object):
  """
  Object-based class for performinhg inference using pytorch and Fastai library.
  """

  def __init__(self, path):
    """
    Initializes class.
    args:
      path --> type: PosixPath
      - working directory, where pytorch weights are located.
    """
    print("Configuring Model")
    self.path = Path(path)
    self.model = self.inference_learner()
    #sets to evaluation mode
    self.model.model.eval()

  @property
  def torch_weights(self):
    """
    Returns algorithm state dictionary for cpu inference.
    """
    return self.path/'app/models/e-covidnet.pth'


  def inference_learner(self):
    """
    Loads an instance of the fastai Learner class with model weights for inference.
    Learner class documentation: https://docs.fast.ai/tutorial.inference.html
    """
    fastai_learner = load_learner(
      path=self.path,
      #fastai settingss (transforms)
      file=self.path/'app/models/e-covidnet.pkl')
    #loads mmodel weights on available device.
    if torch.cuda.is_available():
      fastai_learner.model.load_state_dict(
        torch.load(self.torch_weights))
    else:
      fastai_learner.model.load_state_dict(
        torch.load(self.torch_weights,
          map_location=torch.device("cpu")))
    #Discussion on memory management with Float Tensors vs. Half-Precision Tensors: https://forums.fast.ai/t/comparision-between-to-fp16-and-to-fp32-with-mnist-sample-on-rtx-2070/35693/2
    return fastai_learner.to_fp32()