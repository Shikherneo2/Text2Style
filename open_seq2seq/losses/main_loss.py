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
    self.constant_scale = 1


  def _compute_loss(self, input_dict):
    """
    Computes loss.

    Args:
      input_dict (dict):
       
    Returns:
      Singleton loss tensor
    """
    # sim_matrix_scale = tf.get_variable(
        # name="SimilarityMatrixScale",
        # dtype=self.params['dtype'],
        # trainable=True,
        # initializer=[10.0],
        # constraint=lambda x: tf.clip_by_value(x, 0.5, 25)
    # )
    
    # sim_matrix_bias = tf.get_variable(
        # name="SimilarityMatrixBias",
        # dtype=self.params['dtype'],
        # trainable=True,
        # initializer=[0.0]
    # )

    predicted_token_weights = input_dict["output"]
    target_token_weights = input_dict["target_tensors"][0]
    speaker_ids = input_dict["target_tensors"][1]
    style_embeddings = input_dict["style_embeddings"]
    
    speaker_embedding1 = style_embeddings[0]
    speaker_embedding2 = style_embeddings[1]

    normalized_speaker_embedding1 = tf.math.l2_normalize( speaker_embedding1, axis=-1 )
    normalized_speaker_embedding2 = tf.math.l2_normalize( speaker_embedding2, axis=-1 )
    normalized_speaker_embeddings = tf.concat( [normalized_speaker_embedding1, normalized_speaker_embedding2], 0 )

    speaker_centroid1 = tf.expand_dims( tf.reduce_mean( normalized_speaker_embedding1, axis=0 ), 0 )
    speaker_centroid2 = tf.expand_dims( tf.reduce_mean( normalized_speaker_embedding2, axis=0 ), 0 )

    speaker_centroids = tf.transpose( tf.concat( [speaker_centroid1, speaker_centroid2], axis=0) )

    # Check dimensions for dot product
    # should be batch_size x number of speakers
    similarity_matrix = tf.tensordot( normalized_speaker_embeddings, speaker_centroids, axes=1 )
    #similarity_matrix = tf.clip_by_value(( sim_matrix_scale *similarity_matrix ) + sim_matrix_bias, clip_value_min=0, clip_value_max=1)

    speaker_prob = tf.nn.softmax( similarity_matrix )
    speaker_labels1 = tf.expand_dims( tf.concat( [tf.ones( tf.shape(speaker_embedding1)[0] ), tf.zeros( tf.shape(speaker_embedding2)[0] )], 0 ), 0 )
    speaker_labels2 = 1-speaker_labels1
    speaker_labels = tf.transpose( tf.concat( [speaker_labels1, speaker_labels2], 0 ) )

    loss1 = self.constant_scale*tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels=speaker_labels, logits=speaker_prob ) )

    loss2 = tf.losses.absolute_difference(
        labels=target_token_weights,
        predictions=predicted_token_weights
    )
    # loss2 = tf.losses.mean_squared_error(
                # labels = target_token_weights,
                # predictions = predicted_token_weights
            # )

    loss = loss1+loss2
    return loss
