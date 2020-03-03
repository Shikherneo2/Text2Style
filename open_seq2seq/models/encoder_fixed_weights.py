# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf

from open_seq2seq.models.model import Model
from open_seq2seq.utils.utils import deco_print

class EncoderFixedWeights(Model):
  """
  Calls encoder, concats with fixed length mfcc-coefficients and add another dense layer. Also adds loss.
  """

  @staticmethod
  def get_required_params():
    return dict(Model.get_required_params(), **{
        'encoder': None,  # could be any user defined class
    })

  @staticmethod
  def get_optional_params():
    return dict(Model.get_optional_params(), **{
        'encoder_params': dict,
        'loss': None,  # could be any user defined class
        'loss_params': dict,
    })

  def __init__(self, params, mode="train", hvd=None):
   
    super(EncoderFixedWeights, self).__init__(params=params, mode=mode, hvd=hvd)

    if 'encoder_params' not in self.params:
      self.params['encoder_params'] = {}
   
    if 'loss_params' not in self.params:
      self.params['loss_params'] = {}

    self._encoder = self._create_encoder()
    if self.mode == 'train' or self.mode == 'eval':
      self._loss_computator = self._create_loss()
    else:
      self._loss_computator = None

    self.regularizer = self.params.get('regularizer', None)


  def _create_encoder(self):
    """This function should return encoder class.
    Overwrite this function if additional parameters need to be specified for
    encoder, besides provided in the config.

    Returns:
      instance of a class derived from :class:`encoders.encoder.Encoder`.
    """
    params = self.params['encoder_params']
    return self.params['encoder'](params=params, mode=self.mode, model=self)

  def _create_loss(self):
    """This function should return loss class.
    Overwrite this function if additional parameters need to be specified for
    loss, besides provided in the config.

    Returns:
      instance of a class derived from :class:`losses.loss.Loss`.
    """
    return self.params['loss'](params=self.params['loss_params'], model=self)

  def _build_forward_pass_graph(self, input_tensors, gpu_id=0):
    """TensorFlow graph for encoder-decoder-loss model is created here.
    This function connects encoder, decoder and loss together. As an input for
    encoder it will specify source tensors (as returned from
    the data layer). As an input for decoder it will specify target tensors
    as well as all output returned from encoder. For loss it
    will also specify target tensors and all output returned from
    decoder. Note that loss will only be built for mode == "train" or "eval".

    Args:
      input_tensors (dict): ``input_tensors`` dictionary that has to contain
          ``source_tensors`` key with the list of all source tensors, and
          ``target_tensors`` with the list of all target tensors. Note that
          ``target_tensors`` only need to be provided if mode is
          "train" or "eval".
      gpu_id (int, optional): id of the GPU where the current copy of the model
          is constructed. For Horovod this is always zero.

    Returns:
      tuple: tuple containing loss tensor as returned from
      ``loss.compute_loss()`` and list of outputs tensors, which is taken from
      ``decoder.decode()['outputs']``. When ``mode == 'infer'``, loss will
      be None.
    """
    if not isinstance(input_tensors, dict) or \
       'source_tensors' not in input_tensors:
      raise ValueError('Input tensors should be a dict containing '
                       '"source_tensors" key')

    if not isinstance(input_tensors['source_tensors'], list):
      raise ValueError('source_tensors should be a list')

    source_tensors = input_tensors['source_tensors']
    if self.mode == "train" or self.mode == "eval":
      if 'target_tensors' not in input_tensors:
        raise ValueError('Input tensors should contain "target_tensors" key'
                         'when mode != "infer"')
      if not isinstance(input_tensors['target_tensors'], list):
        raise ValueError('target_tensors should be a list')
      target_tensors = input_tensors['target_tensors']

    with tf.variable_scope("ForwardPass"):
      encoder_input = {"source_tensors": source_tensors}
      encoder_output = self.encoder.encode(input_dict=encoder_input)

      top_layer = encoder_output
      # top_layer = tf.layers.dense(
      #       encoder_output,
      #       256,
      #       activation=tf.nn.tanh,
      #       kernel_regularizer=self.regularizer,
      #       name="text_encoder_activation"
      # )

      # Expect MFCC tensor to be at the end of the source tensors
      style_embedding = input_tensors['source_tensors'][2]
      
      # dense layer from raw mfcc data
      dense_outputs_style = tf.layers.dense(
            style_embedding,
            self.params["mfcc_dims"],
            activation=tf.nn.relu,
            kernel_regularizer=self.regularizer,
            name="concatenated_encoder_activation"
      )
      outputs = tf.concat([top_layer, dense_outputs_style], axis=-1)

      # pass the concatenated tensors through a dense layer outputing the final dimension
      dense_outputs = tf.layers.dense(
            outputs,
            512,
            activation=tf.nn.tanh,
            kernel_regularizer=self.regularizer,
            name="concatenated_encoder_activation"
      )

      if self.mode == "train" or self.mode == "eval":
        with tf.variable_scope("Loss"):
          loss_input_dict = {
              "output": dense_outputs,
              "target_tensors": target_tensors,
          }
          loss = self.loss_computator.compute_loss(loss_input_dict)
      else:
        deco_print("Inference Mode. Loss part of graph isn't built.")
        loss = None
      
      # Add model_outputs
      # model_outputs = decoder_output.get("outputs", None)
      model_outputs = [dense_outputs]
      return loss, model_outputs

  @property
  def encoder(self):
    """Model encoder."""
    return self._encoder

  @property
  def loss_computator(self):
    """Model loss computator."""
    return self._loss_computator
