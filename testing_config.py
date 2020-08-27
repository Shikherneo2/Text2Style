# pylint: skip-file
import os
import tensorflow as tf
from open_seq2seq.models import Text2Style
from open_seq2seq.encoders import Tacotron2Encoder
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import MainLoss
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay


base_location = "/home/sdevgupta/mine/Text2Style"
dataset_location = os.path.join( base_location, "open_seq2seq/dataset" )
logdir = os.path.join( base_location, "speaker_verification_scaled/" )

batch_size = 90
base_model = Text2Style

# Only set this to True in inference if the model being used was trained with this turned on
use_minimal_vocabulary = True
saved_embedding_location_train = "/home/sdevgupta/mine/OpenSeq2Seq/ljspeech_catheryn_logs/embeddings"
saved_embedding_location_val = "/home/sdevgupta/mine/OpenSeq2Seq/ljspeech_catheryn_logs/embeddings_val2"

train = os.path.join(dataset_location, "combined_dataset_for_text2style_training.csv")
val = os.path.join(dataset_location, "local_combined_dataset_val2.csv")
infer = "/home/sdevgupta/mine/Text2Style/open_seq2seq/dataset/test_sentences_lj_combined.csv"


if use_minimal_vocabulary:
    vocab_file_location = "open_seq2seq/test_utils/minimal_vocab_tts.txt"
else:
    vocab_file_location = "open_seq2seq/test_utils/reduced_vocab_tts.txt"

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 1,
  "num_epochs": 5000,

  "batch_size_per_gpu": batch_size,

  "save_summaries_steps": 150,
  "print_loss_steps": 150,
  "print_samples_steps": None,
  "eval_steps": 500,
  "save_checkpoint_steps": 5000,
  "save_to_tensorboard": True,
  "logdir": logdir,
  "max_grad_norm":1.,

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-4,
    "decay_steps": 20000,
    "decay_rate": 0.5,
    "use_staircase_decay": False,
    "begin_decay_at": 110000,
    "min_lr": 5e-6,
  },
  "dtype": tf.float32,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 1e-4
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'gradient_norm', 'global_gradient_norm', 'variable_norm'],

  "encoder": Tacotron2Encoder,
  "encoder_params": {
    "cnn_dropout_prob": 0.5,
    "rnn_dropout_prob": 0.5,
    'src_emb_size': 512,
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
    ],
    "activation_fn": tf.nn.relu,

    "num_rnn_layers": 1,
    "rnn_cell_dim": 256,
    "rnn_unidirectional": False,
    "use_cudnn_rnn": False,
    #"rnn_type": tf.contrib.cudnn_rnn.CudnnGRU,
    "rnn_type": tf.nn.rnn_cell.GRUCell,
    "zoneout_prob": 0.,

    "data_format": "channels_last",

    "style_embedding_enable": True,
    "style_embedding_params": {
      "conv_layers": [
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 32, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 32, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 64, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 64, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 128, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 128, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 256, "padding": "SAME"
        }
      ],
      "num_rnn_layers": 1,
      "rnn_cell_dim": 256,
      "rnn_unidirectional": False,
      #"rnn_type": tf.contrib.cudnn_rnn.CudnnGRU,
      "rnn_type": tf.nn.rnn_cell.GRUCell,
      "emb_size": 512,
    }
  },
  
  "loss": MainLoss,
  "loss_params": {
    "use_mask": False
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
        "saved_embedding_location_train": saved_embedding_location_train,
        "saved_embedding_location_val": saved_embedding_location_val,
        "mel_feature_num": 80,
        "vocab_file": vocab_file_location,
        'dataset_location': dataset_location,
        "minimal_vocabulary": use_minimal_vocabulary,
        "pad_EOS": True,
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      train,
    ],
    "shuffle": True,
    "style_input": "wav"
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      val,
    ],
    "duration_max":10000,
    "duration_min":0,
    "shuffle": False,
    "style_input": "wav"
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
        infer
  ],
    "duration_max":10000,
    "duration_min":0,
    "shuffle": False,
    "style_input": "wav"
  },
}
