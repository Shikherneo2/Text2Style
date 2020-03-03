# Copyright (c) 2019 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from encoder_fixed_weights import EncoderFixedWeights


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

  def print_logs(self,
                 mode,
                 specs,
                 titles,
                 stop_token_pred,
                 stop_target,
                 audio_length,
                 step,
                 predicted_final_spec,
                 predicted_mag_spec=None):
    """
    Save audio files and plots.

    Args:
      mode: "train" or "eval".
      specs: spectograms to plot.
      titles: spectogram titles.
      stop_token_pred: stop token prediction.
      stop_target: stop target.
      audio_length: length of the audio.
      step: current step.
      predicted_final_spec: predicted mel spectogram.
      predicted_mag_spec: predicted magnitude spectogram.

    Returns:
      Dictionary to log.
    """

    dict_to_log = {}

    if self._save_to_tensorboard:
      save_format = "tensorboard"
    else:
      save_format = "disk"


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
      input_values = sample[0]["source_tensors"][2]
      outputs = sample[1][0]
      for j in len(input_values):
        np.save( os.path.join(self.params["logdir"], os.path.basename(input_values[j]) ), outputs[j] )


  def finalize_evaluation(self, results_per_batch, training_step=None, samples_count=1):
    print( "Evaluation not implemented yet." )
    return []


  def maybe_print_logs(self, input_values, output_values, training_step):
    spec, stop_target, _=input_values['target_tensors']
    predicted_decoder_spec=output_values[0]
    predicted_final_spec=output_values[1]
    attention_mask=output_values[2]
    stop_token_pred=output_values[3]
    y_sample=spec[0]
    stop_target=stop_target[0]
    predicted_spec=predicted_decoder_spec[0]
    predicted_final_spec=predicted_final_spec[0]
    alignment=attention_mask[0]
    stop_token_pred=stop_token_pred[0]
    audio_length=output_values[4][0]

    specs=[
        y_sample,
        predicted_spec,
        predicted_final_spec
    ]

    titles=[
        "training data",
        "decoder results",
        "post net results"
    ]

    alignment_specs, alignment_titles=self.get_alignments(alignment)
    specs += alignment_specs
    titles += alignment_titles

    predicted_mag_spec=None

    if "both" in self.get_data_layer().params["output_type"]:
      predicted_mag_spec=output_values[5][0]
      specs.append(predicted_mag_spec)
      titles.append("magnitude spectrogram")
      n_feats=self.get_data_layer().params["num_audio_features"]
      mel, mag=np.split(
          y_sample,
          [n_feats["mel"]],
          axis=1
      )
      specs.insert(0, mel)
      specs[1]=mag
      titles.insert(0, "target mel")
      titles[1]="target mag"

    return self.print_logs(
        mode="train",
        specs=specs,
        titles=titles,
        stop_token_pred=stop_token_pred,
        stop_target=stop_target,
        audio_length=audio_length,
        step=training_step,
        predicted_final_spec=predicted_final_spec,
        predicted_mag_spec=predicted_mag_spec
    )
