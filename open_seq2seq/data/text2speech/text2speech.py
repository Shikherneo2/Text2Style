""" # Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
"""

import os
import six
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

from six import string_types

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary

class Text2SpeechDataLayer(DataLayer):
  """
  Text-to-speech data layer class
  """

  @staticmethod
  def get_required_params():
    return dict(
        DataLayer.get_required_params(), **{
            'dataset_location': str,
            'mel_feature_num': None,
            'vocab_file': str,
            'dataset_files': list,
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        DataLayer.get_optional_params(), **{
            'pad_to': int,
            'mag_power': int,
            'pad_EOS': bool,
            'pad_value': float,
            'feature_normalize_mean': float,
            'feature_normalize_std': float,
            'trim': bool,
            'data_min': None,
            'duration_min': int,
            'duration_max': int,
            'mel_type': ['slaney', 'htk'],
            "exp_mag": bool,
            'style_input': [None, 'wav'],
            'n_samples_train': int,
            'n_samples_eval': int,
            'n_fft': int,
            'fmax': float,
            'max_normalization': bool,
            'use_cache': bool,
            
            "saved_embedding_location_train": str,
            "saved_embedding_location_val": str,
        }
    )


  def __init__(self, params, model, num_workers=None, worker_id=None):
    super(Text2SpeechDataLayer, self).__init__(
        params,
        model,
        num_workers,
        worker_id
    )

    self.use_cache = self.params.get('use_cache', False)
    self._cache = {}

    names = [ 'mel_file', 'transcript', "fileid" ]
    sep = '\x7c'
    header = None

    # Character level vocab
    self.params['char2idx'] = load_pre_existing_vocabulary(
        self.params['vocab_file'],
        min_idx=3,
        read_chars=True,
    )
    # Add the pad, start, and end chars
    self.params['char2idx']['<p>'] = 0
    self.params['char2idx']['<s>'] = 1
    self.params['char2idx']['</s>'] = 2
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    self.params['src_vocab_size'] = len(self.params['char2idx'])

    # Load csv files
    self._files = None
    for csvs in params['dataset_files']:
      files = pd.read_csv(
          csvs,
          encoding='utf-8',
          sep=sep,
          header=header,
          names=names,
          quoting=3
      )
      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)

    if self.params['mode'] == 'train' and 'n_samples_train' in self.params:
      indices = self._files['transcript'].str.len().sort_values().index
      self._files = self._files.reindex(indices)

      n_samples = self.params.get('n_samples_train')
      print('Using just the {} shortest samples'.format(n_samples))
      self._files = self._files.iloc[:n_samples]

    if self.params['mode'] == 'eval':
      indices = self._files['transcript'].str.len().sort_values().index
      self._files = self._files.reindex(indices)

      if 'n_samples_eval' in self.params:
        n_samples = self.params['n_samples_eval']
        self._files = self._files.iloc[:n_samples]

    cols = ['mel_file', 'transcript', "fileid"]

    all_files = self._files.loc[:, cols].values
    all_files = [ list(map(str, i)) for i in all_files ]
    self._files = self.split_data(all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

  def split_data(self, data):
    if self.params['mode'] != 'train' and self._num_workers is not None:
      size = len(data)
      start = size // self._num_workers * self._worker_id
      if self._worker_id == self._num_workers - 1:
        end = size
      else:
        end = size // self._num_workers * (self._worker_id + 1)
      return data[start:end]
    return data


  @property
  def iterator(self):
    return self._iterator


  def build_graph(self):
    with tf.device('/cpu:0'):
      """Builds data reading graph."""

      self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
      if self.params['shuffle']:
        self._dataset = self._dataset.shuffle(self._size)
      self._dataset = self._dataset.repeat()

      # txt, txt_length, mfcc, token_weights
      pad_value = np.log(self.params.get("data_min", 1e-5))
      
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self._parse_spec_embed_element,
              [line],
              [ tf.int32, tf.int32, self.params['dtype'], tf.int32, self.params['dtype'], self.params['dtype'], self.params['dtype'], tf.int32 ],
              stateful=False,
          ),
          num_parallel_calls=1,
      )

      self._dataset = self._dataset.filter(
          lambda text, text_length, spectrogram, spec_length, style_embedding, words_per_frame, chars_per_frame, fileid:
                tf.greater_equal(
                  spec_length,
                  120
                )
      )

      self._dataset = self._dataset.padded_batch(
          self.params['batch_size'],

          padded_shapes=(
              [None], 1, [None, self.params["mel_feature_num"]], [None], 512, 1, 1, 1
          ),
          padding_values=(
              0, 0, tf.cast(pad_value, dtype=self.params['dtype']), 0, tf.cast(0, dtype=self.params['dtype']), tf.cast(0, dtype=self.params['dtype']), tf.cast(0, dtype=self.params['dtype']),  0
          )
      )

      self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE).make_initializable_iterator()

      text, text_length, spectrogram, spec_length, style_embedding, words_per_frame, chars_per_frame, fileid = self._iterator.get_next()

      # Reshape tensors along batch size
      text.set_shape([self.params['batch_size'], None])
      text_length = tf.reshape(text_length, [self.params['batch_size']])
      spectrogram.set_shape([self.params['batch_size'], None, self.params["mel_feature_num"]])
      spec_length = tf.reshape(spec_length, [self.params['batch_size']])
      style_embedding.set_shape([self.params['batch_size'], 512])
      words_per_frame = tf.reshape(words_per_frame, [self.params['batch_size'],1])
      chars_per_frame = tf.reshape(chars_per_frame, [self.params['batch_size'],1])

      self._input_tensors = {}
      self._input_tensors["source_tensors"] = [text, text_length, spectrogram, spec_length, words_per_frame, chars_per_frame, fileid]
      
      if self.params['mode'] != 'infer':
        self._input_tensors['target_tensors'] = [ style_embedding ]


  def _parse_spec_embed_element(self, element):
    mfcc_filename, transcript, fileid = element
    transcript = transcript.lower()
    
    if six.PY2:
      mel_filename = unicode( mfcc_filename, "utf-8" )
      transcript = unicode( transcript, "utf-8" )
      fileid = unicode( fileid, "utf-8" )

    elif not isinstance(transcript, string_types):
      mel_filename = str( mfcc_filename, "utf-8" )
      transcript = str( transcript, "utf-8" )
      embedding_filename = str( fileid, "utf-8" )

    if self.params['mode'] == 'eval':
      embedding_filename = os.path.join( self.params["saved_embedding_location_val"], "embed-"+embedding_filename+".npy" )
    elif self.params['mode'] == 'train':
      embedding_filename = os.path.join( self.params["saved_embedding_location_train"], "embed-"+embedding_filename+".npy" )
    
    transcript = transcript.upper()
    normalized_transcript = transcript.replace(","," ")
    num_words = len([i for i in normalized_transcript.split(" ") if i.strip()!=""])
    
    text_input = np.array(
      [self.params['char2idx'][c] for c in transcript]
    )

    num_chars = text_input.shape[0]
    # Do we still need to pad if we compress the text into fixed length output vectors?
    pad_to = self.params.get('pad_to', 8)
    if self.params.get("pad_EOS", True):
      num_pad = pad_to - ((len(text_input) + 2) % pad_to)
      text_input = np.pad(
        text_input, ((1, 1)),
        "constant",
        constant_values=(
            (self.params['char2idx']["<s>"], self.params['char2idx']["</s>"])
        )
      )
      text_input = np.pad(
        text_input, ((0, num_pad)),
        "constant",
        constant_values=self.params['char2idx']["<p>"]
      )
  
  # saved mels are of shape 80,num_frames
    mel_filename = os.path.join("/home/sdevgupta/mine/data/mels_ground_truth", "_".join(mel_filename.split("/")[-2:]) + ".npy")
    mel = np.load(mel_filename).T
    mel_length = float(mel.shape[0])
    num_pad = pad_to - ((len(mel) + 1) % pad_to) + 1
    
    mel = np.pad(
            mel,
            ((0, num_pad), (0, 0)),
            "constant",
            constant_values=np.log(1e-5)
          )

    if self.params["mode"] == "infer":
      # Dummy embedding. infer does not need this.
      style_embedding = np.zeros(512)
    else:
      style_embedding = np.squeeze( np.load(embedding_filename) )

    if self.params['mode'] == 'infer':
      assoc = [ "a_room_with_a_view_01-000099.wav", "a_little_princess_11-000090.wav", "daisy_miller_01-000031.wav", "daisy_miller_02-000239.wav", "emma_03-000004.wav", "emma_03-000058.wav", "through_the_looking_glass_01-000058.wav", "mansfield_park_02-000108.wav", "mansfield_park_05-000042.wav"]
      
      words_per_mel_frame = 0.08824324
      chars_per_mel_frame = 0.45500421
    else:
      words_per_mel_frame = num_words/mel_length
      chars_per_mel_frame = num_chars/mel_length

    words_per_mel_frame = num_words/mel_length
    chars_per_mel_frame = num_chars/mel_length

    assert len(text_input) % pad_to == 0
    assert mel.shape[1] == self.params["mel_feature_num"]
    assert style_embedding.shape[0] == 512

    return np.int32( text_input ), \
           np.int32( [len(text_input)] ), \
           mel.astype( np.float32 ), \
           np.int32( [len(mel)] ), \
           style_embedding.astype( np.float32 ), \
           np.float32( [ words_per_mel_frame  ] ), \
           np.float32( [ chars_per_mel_frame  ] ), \
           np.int32( [int(fileid)] )  


  @property
  def input_tensors(self):
    return self._input_tensors

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)
