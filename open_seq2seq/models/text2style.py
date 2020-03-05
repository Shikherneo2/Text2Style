# Copyright (c) 2019 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from open_seq2seq.models.encoder_fixed_weights import EncoderFixedWeights


class Text2Style(EncoderFixedWeights):
  """
  Text-to-speech data layer.
  """

  @staticmethod
  def get_required_params():
    return dict(
        EncoderFixedWeights.get_required_params(), **{
            "save_to_tensorboard": bool,
        }
    )

  def __init__(self, params, mode="train", hvd=None):
    super(Text2Style, self).__init__(params, mode=mode, hvd=hvd)
    self._save_to_tensorboard = self.params["save_to_tensorboard"]


  def infer(self, input_values, output_values):
    if self.on_horovod:
      raise ValueError("Inference is not supported on horovod")

    return [input_values, output_values]

  def evaluate(self, input_values, output_values):
    # Need to reduce amount of data sent for horovod
    # Use last element
    idx = -1
    output_values = [(item[idx]) for item in output_values]
    input_values = {
        key: [value[0][idx], value[1][idx]] for key, value in input_values.items()
    }
    return [input_values, output_values]


  def finalize_inference(self, results_per_batch, output_file):
    print("output_file is ignored for tts")
    print("results are logged to the logdir")

    for i, sample in enumerate(results_per_batch):
      input_values = sample[0]["source_tensors"][0]
      outputs = sample[1][0]
      for j in range(len(input_values)):
        h = os.path.join(self.params["logdir"], "outputs", "mel-"+str(j)+".npy" )
        np.save( h, outputs[j] )


  def finalize_evaluation(self, results_per_batch, training_step=None, samples_count=1):
    print( "Evaluation not implemented yet." )
    return []


  def maybe_print_logs(self, input_values, output_values, training_step):
    return []
