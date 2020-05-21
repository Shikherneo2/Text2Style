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
logdir = os.path.join( base_location, "logs_mixed_phoneme_tacotron" )
saved_embedding_location = "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/embeddings"
vocab_file_location = "open_seq2seq/test_utils/reduced_vocab_tts.txt"

batch_size = 64
base_model = Text2Style
saved_embedding_location_train = "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/train_text2_style_dataset_60K_single_batch"
saved_embedding_location_val = "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/val_text2_style_dataset_60K_single_batch"

train = os.path.join(dataset_location, "embeddings_infer_dataset_for_training_local.csv")
val = os.path.join(dataset_location, "embeddings_infer_dataset_for_validation_local.csv")
infer = "/home/sdevgupta/mine/OpenSeq2Seq/dataset/test_sentences.csv"

base_params = {
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 1,
  "num_epochs": 100,

  "batch_size_per_gpu": batch_size,

  "save_summaries_steps": 100,
  "print_loss_steps": 100,
  "print_samples_steps": None,
  "eval_steps": 500,
  "save_checkpoint_steps": 2500,
  "save_to_tensorboard": True,
  "logdir": logdir,
  "max_grad_norm":1.,

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-4,
    "decay_steps": 5000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 20000,
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
