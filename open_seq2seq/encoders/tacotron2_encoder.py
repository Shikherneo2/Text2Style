# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import numpy as np
import inspect

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.framework import ops
from open_seq2seq.parts.cnns.conv_blocks import conv_bn_actv
from open_seq2seq.parts.rnns.utils import single_cell

from .encoder import Encoder


class Tacotron2Encoder(Encoder):
  """Tacotron-2 like encoder.

  Consists of an embedding layer followed by a convolutional layer followed by
  a recurrent layer.
  """

  @staticmethod
  def get_required_params():
    return dict(
        Encoder.get_required_params(),
        **{
            'cnn_dropout_prob': float,
            'rnn_dropout_prob': float,
            'src_emb_size': int,
            'conv_layers': list,
            'activation_fn': None,  # any valid callable
            'num_rnn_layers': int,
            'rnn_cell_dim': int,
            'use_cudnn_rnn': bool,
            'rnn_type': None,
            'rnn_unidirectional': bool,
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        Encoder.get_optional_params(), **{
            'data_format': ['channels_first', 'channels_last'],
            'bn_momentum': float,
            'bn_epsilon': float,
            'zoneout_prob': float,
            'style_embedding_enable': bool,
            'style_embedding_params': dict,
        }
    )

  def __init__(self, params, model, name="tacotron2_encoder", mode='train'):
    """Tacotron-2 like encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **cnn_dropout_prob** (float) --- dropout probabilty for cnn layers.
    * **rnn_dropout_prob** (float) --- dropout probabilty for cnn layers.
    * **src_emb_size** (int) --- dimensionality of character embedding.
    * **conv_layers** (list) --- list with the description of convolutional
      layers. For example::
        "conv_layers": [
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          },
          {
            "kernel_size": [5], "stride": [1],
            "num_channels": 512, "padding": "SAME"
          }
        ]
    * **activation_fn** (callable) --- activation function to use for conv
      layers.
    * **num_rnn_layers** --- number of RNN layers to use.
    * **rnn_cell_dim** (int) --- dimension of RNN cells.
    * **rnn_type** (callable) --- Any valid RNN Cell class. Suggested class is
      lstm
    * **rnn_unidirectional** (bool) --- whether to use uni-directional or
      bi-directional RNNs.
    * **zoneout_prob** (float) --- zoneout probability. Defaults to 0.
    * **use_cudnn_rnn** (bool) --- need to be enabled in rnn_type is a Cudnn
      class.
    * **data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.1.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-5.
    * **style_embedding_enable** (bool) --- Whether to enable GST. Defaults to
      False.
    * **style_embedding_params** (dict) --- Parameters for GST layer. See
      _embed_style documentation.
    """
    super(Tacotron2Encoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """Creates TensorFlow graph for Tacotron-2 like encoder.

    Args:
       input_dict (dict): dictionary with inputs.
        Must define:

            source_tensors - array containing [

              * source_sequence: tensor of shape [batch_size, sequence length]
              * src_length: tensor of shape [batch_size]

            ]

    Returns:
      dict: A python dictionary containing:

          * outputs - tensor containing the encoded text to be passed to the
            attention layer
          * src_length - the length of the encoded text
    """

    text = input_dict['source_tensors'][0]
    text_len = input_dict['source_tensors'][1]
    mel = input_dict['source_tensors'][2]
    mel_length = input_dict['source_tensors'][3]
    words_per_frame = input_dict['source_tensors'][4]
    chars_per_frame = input_dict['source_tensors'][5]

    batch_size = mel.get_shape().as_list()[0]
    training = (self._mode == "train")
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_last')
    src_vocab_size = self._model.get_data_layer().params['src_vocab_size']
    zoneout_prob = self.params.get('zoneout_prob', 0.)

    if training:
        speaker_id = input_dict['target_tensors'][1]
    else:
        speaker_id = tf.ones((batch_size,1), dtype=tf.float32)
    # if src_vocab_size % 8 != 0:
    #   src_vocab_size += 8 - (src_vocab_size % 8)

    # ----- Embedding layer -----------------------------------------------
    enc_emb_w = tf.get_variable(
        name="EncoderEmbeddingMatrix",
        shape=[src_vocab_size, self.params['src_emb_size']],
        dtype=self.params['dtype'],
        # initializer=tf.random_normal_initializer()
    )

    embedded_inputs = tf.cast(
        tf.nn.embedding_lookup(
            enc_emb_w,
            text,
        ), self.params['dtype']
    )

    # ----- Convolutional layers -----------------------------------------------
    input_layer = embedded_inputs

    if data_format == 'channels_last':
      top_layer = input_layer
    else:
      top_layer = tf.transpose(input_layer, [0, 2, 1])

    for i, conv_params in enumerate(self.params['conv_layers']):
      ch_out = conv_params['num_channels']
      kernel_size = conv_params['kernel_size']  # [time, freq]
      strides = conv_params['stride']
      padding = conv_params['padding']

      if padding == "VALID":
        text_len = (text_len - kernel_size[0] + strides[0]) // strides[0]
      else:
        text_len = (text_len + strides[0] - 1) // strides[0]

      top_layer = conv_bn_actv(
          layer_type="conv1d",
          name="conv{}".format(i + 1),
          inputs=top_layer,
          filters=ch_out,
          kernel_size=kernel_size,
          activation_fn=self.params['activation_fn'],
          strides=strides,
          padding=padding,
          regularizer=regularizer,
          training=training,
          data_format=data_format,
          bn_momentum=self.params.get('bn_momentum', 0.1),
          bn_epsilon=self.params.get('bn_epsilon', 1e-5),
      )
      top_layer = tf.layers.dropout(
          top_layer, rate=self.params["cnn_dropout_prob"], training=training
      )

    if data_format == 'channels_first':
      top_layer = tf.transpose(top_layer, [0, 2, 1])

    # ----- RNN ---------------------------------------------------------------
    num_rnn_layers = self.params['num_rnn_layers']
    cell_params = {}
    cell_params["num_units"] = self.params['rnn_cell_dim']
    rnn_type = self.params['rnn_type']
    rnn_input = top_layer
    rnn_vars = []

    multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
        [
            single_cell(
                cell_class=rnn_type,
                cell_params=cell_params,
                zoneout_prob=zoneout_prob,
                training=training,
                residual_connections=False
            ) for _ in range(num_rnn_layers)
        ]
    )
    rnn_vars += multirnn_cell_fw.trainable_variables

    if self.params['rnn_unidirectional']:
      top_layer, final_state = tf.nn.dynamic_rnn(
          cell=multirnn_cell_fw,
          inputs=rnn_input,
          sequence_length=text_len,
          dtype=rnn_input.dtype,
          time_major=False,
      )
      final_state = final_state[0]
    else:
      multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
          [
              single_cell(
                  cell_class=rnn_type,
                  cell_params=cell_params,
                  zoneout_prob=zoneout_prob,
                  training=training,
                  residual_connections=False
              ) for _ in range(num_rnn_layers)
          ]
      )
      top_layer, final_state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=multirnn_cell_fw,
          cell_bw=multirnn_cell_bw,
          inputs=rnn_input,
          sequence_length=text_len,
          dtype=rnn_input.dtype,
          time_major=False
      )
      # concat 2 tensors [B, T, n_cell_dim] --> [B, T, 2*n_cell_dim]
      # final_state = tf.concat((final_state[0][0], final_state[1][0]), 1)
      top_layer = tf.concat(top_layer, 2)
      rnn_vars += multirnn_cell_bw.trainable_variables

    # -- end of rnn------------------------------------------------------------

    top_layer = tf.layers.dropout(
        top_layer, rate=self.params["rnn_dropout_prob"], training=training
    )
    with tf.variable_scope("style_encoder"):
      style_embedding = self._embed_style( mel, mel_length )
      style_embedding = tf.concat( [style_embedding, words_per_frame, chars_per_frame], axis=-1 )
      style_embedding = tf.concat( [style_embedding, tf.cast(speaker_id, dtype=tf.float32)], axis=-1 )

    speaker_id = tf.squeeze( speaker_id )
    style_embedding = tf.layers.dense(
          style_embedding,
          256,
          activation=tf.nn.tanh,
          kernel_regularizer=regularizer,
          name="extended_style_dense_layer"
      )

    if training:
        style_speaker_wise = []
        indices = tf.where( tf.equal(speaker_id, 0) )
        
        # indices = tf.reshape( indices, tf.shape(indices.shape)[0] )
        indices = tf.squeeze( indices )
        style_speaker_wise1 = tf.gather( style_embedding, indices, axis=0 )
        style_speaker_wise.append( style_speaker_wise1 )

        indices = tf.where( tf.equal(speaker_id, 1) )
        # indices = tf.reshape( indices, tf.shape(indices.shape)[0] )
        indices = tf.squeeze( indices )
        style_speaker_wise2 = tf.gather( style_embedding, indices, axis=0 )
        style_speaker_wise.append( style_speaker_wise2 )

    style_embedding = tf.expand_dims(style_embedding, 1)
    style_embedding = tf.tile(
        style_embedding,
        [1, tf.reduce_max(text_len), 1]
    )

    outputs = tf.concat([top_layer, style_embedding], axis=-1)

    with tf.variable_scope("concatenated_encoder"):
      cell_params = {}
      cell_params["num_units"] = 512
      multirnn_cell_fw2 = tf.nn.rnn_cell.MultiRNNCell(
            [
                single_cell(
                    cell_class=rnn_type,
                    cell_params=cell_params,
                    training=training,
                    residual_connections=False
                ) for _ in range(num_rnn_layers)
            ]
        )
      rnn_vars += multirnn_cell_fw2.trainable_variables
      
      multirnn_cell_bw2 = tf.nn.rnn_cell.MultiRNNCell(
            [
                single_cell(
                    cell_class=rnn_type,
                    cell_params=cell_params,
                    zoneout_prob=zoneout_prob,
                    training=training,
                    residual_connections=False
                ) for _ in range(num_rnn_layers)
            ]
        )

      _, final_state = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=multirnn_cell_fw2,
          cell_bw=multirnn_cell_bw2,
          inputs=outputs,
          sequence_length=text_len,
          dtype=rnn_input.dtype,
          time_major=False
      )

      rnn_vars += multirnn_cell_bw2.trainable_variables
      final_state = tf.concat((final_state[0][0], final_state[1][0]), 1)

    if regularizer and training:
      cell_weights = []
      cell_weights += rnn_vars
      cell_weights += [enc_emb_w]
      for weights in cell_weights:
        if "bias" not in weights.name:
          # print("Added regularizer to {}".format(weights.name))
          if weights.dtype.base_dtype == tf.float16:
            tf.add_to_collection(
                'REGULARIZATION_FUNCTIONS', (weights, regularizer)
            )
          else:
            tf.add_to_collection(
                ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights)
            )

    dense_outputs = tf.layers.dense(
            final_state,
            1024,
            activation=tf.nn.tanh,
            kernel_regularizer=regularizer,
            name="concatenated_encoder_activation1"
    )

    dense_outputs = tf.layers.dense(
            dense_outputs,
            512,
            activation=tf.nn.tanh,
            kernel_regularizer=regularizer,
            name="concatenated_encoder_activation2"
    )
    if training:
        return {
            'outputs': dense_outputs,
            'src_length': text_len,
            'style_embeddings': style_speaker_wise
        }
    else:
        return {
            'outputs': dense_outputs,
            'src_length': text_len,
        }
  

  def _embed_style( self, style_spec, style_len ):
    """
    Code that implements the reference encoder as described in "Towards
    end-to-end prosody transfer for expressive speech synthesis with Tacotron",
    and "Style Tokens: Unsupervised Style Modeling, Control and Transfer in
    End-to-End Speech Synthesis"

    Config parameters:

    * **conv_layers** (list) --- See the conv_layers parameter for the
      Tacotron-2 model.
    * **num_rnn_layers** (int) --- Number of rnn layers in the reference encoder
    * **rnn_cell_dim** (int) --- Size of rnn layer
    * **rnn_unidirectional** (bool) --- Uni- or bi-directional rnn.
    * **rnn_type** --- Must be a valid tf rnn cell class
    * **emb_size** (int) --- Size of gst
    * **attention_layer_size** (int) --- Size of linear layers in attention
    * **num_tokens** (int) --- Number of tokens for gst
    * **num_heads** (int) --- Number of attention heads
    """
    batch_size = style_spec.get_shape().as_list()[0]
    
    training = (self._mode == "train")
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_last')

    top_layer = tf.expand_dims(style_spec, -1)
    params = self.params['style_embedding_params']
    if "conv_layers" in params:
      for i, conv_params in enumerate(params['conv_layers']):
        ch_out = conv_params['num_channels']
        kernel_size = conv_params['kernel_size']  # [time, freq]
        strides = conv_params['stride']
        padding = conv_params['padding']

        if padding == "VALID":
          style_len = (style_len - kernel_size[0] + strides[0]) // strides[0]
        else:
          style_len = (style_len + strides[0] - 1) // strides[0]

        top_layer = conv_bn_actv(
            layer_type="conv2d",
            name="conv{}".format(i + 1),
            inputs=top_layer,
            filters=ch_out,
            kernel_size=kernel_size,
            activation_fn=self.params['activation_fn'],
            strides=strides,
            padding=padding,
            regularizer=regularizer,
            training=training,
            data_format=data_format,
            bn_momentum=self.params.get('bn_momentum', 0.1),
            bn_epsilon=self.params.get('bn_epsilon', 1e-5),
        )

      if data_format == 'channels_first':
        top_layer = tf.transpose(top_layer, [0, 2, 1])

    top_layer = tf.concat(tf.unstack(top_layer, axis=2), axis=-1)
    top_layer = tf.layers.dropout(
          top_layer, rate=self.params["cnn_dropout_prob"], training=training
      )

    num_rnn_layers = params['num_rnn_layers']
    if num_rnn_layers > 0:
      cell_params = {}
      cell_params["num_units"] = params['rnn_cell_dim']
      rnn_type = params['rnn_type']
      rnn_input = top_layer
      rnn_vars = []

      multirnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
          [
              single_cell(
                  cell_class=rnn_type,
                  cell_params=cell_params,
                  training=training,
                  residual_connections=False
              ) for _ in range(num_rnn_layers)
          ]
      )
      rnn_vars += multirnn_cell_fw.trainable_variables
      if params['rnn_unidirectional']:
        top_layer, final_state = tf.nn.dynamic_rnn(
            cell=multirnn_cell_fw,
            inputs=rnn_input,
            sequence_length=style_len,
            dtype=rnn_input.dtype,
            time_major=False,
        )
        final_state = final_state[0]
      else:
        multirnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [
                single_cell(
                    cell_class=rnn_type,
                    cell_params=cell_params,
                    training=training,
                    residual_connections=False
                ) for _ in range(num_rnn_layers)
            ]
        )
        top_layer, final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=multirnn_cell_fw,
            cell_bw=multirnn_cell_bw,
            inputs=rnn_input,
            sequence_length=style_len,
            dtype=rnn_input.dtype,
            time_major=False
        )
        # concat 2 tensors [B, T, n_cell_dim] --> [B, T, 2*n_cell_dim]
        final_state = tf.concat((final_state[0][0], final_state[1][0]), 1)
        rnn_vars += multirnn_cell_bw.trainable_variables

      top_layer = final_state
      # Apply linear layer
      top_layer = tf.layers.dense(
          top_layer,
          256,
          activation=tf.nn.tanh,
          kernel_regularizer=regularizer,
          name="reference_activation"
      )

      if regularizer and training:
        cell_weights = rnn_vars
        for weights in cell_weights:
          if "bias" not in weights.name:
            # print("Added regularizer to {}".format(weights.name))
            if weights.dtype.base_dtype == tf.float16:
              tf.add_to_collection(
                  'REGULARIZATION_FUNCTIONS', (weights, regularizer)
              )
            else:
              tf.add_to_collection(
                  ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights)
              )

    return top_layer
