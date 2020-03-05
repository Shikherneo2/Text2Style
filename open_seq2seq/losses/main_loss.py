# Copyright (c) 2019 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from .loss import Loss


class MainLoss(Loss):
  """
  Default text-to-speech loss.
  """

  @staticmethod
  def get_optional_params():
    return {
        "use_mask": bool,
    }

  def __init__(self, params, model, name="main_loss"):
    super(MainLoss, self).__init__(params, model, name)


  def _compute_loss(self, input_dict):
    """
    Computes loss.

    Args:
      input_dict (dict):
       
    Returns:
      Singleton loss tensor
    """

    predicted_token_weights = input_dict["output"]
    target_token_weights = input_dict["target_tensors"][0]

    loss = tf.losses.absolute_difference(
        labels=target_token_weights,
        predictions=predicted_token_weights
    )

    return loss
