# pylint: skip-file
import os
import tensorflow as tf
from open_seq2seq.models import Text2Style
from open_seq2seq.encoders import Tacotron2Encoder
from open_seq2seq.decoders import Tacotron2Decoder
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.losses import MainLoss


base_model = Text2Style

dataset_location = "/home/sdevgupta/mine/Text2Style/dataset"

base_params = {
  # save token weights when infering
  "save_token_weights": False,
  "use_weight_files": True,
  "weights_file": "/home/sdevgupta/mine/OpenSeq2Seq/logs4/embed_npys/token_scale_weights.npy",
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 1,
  "num_epochs": 25,

  "batch_size_per_gpu": batch_size,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 500,
  "eval_steps": 500,
  "save_checkpoint_steps": 2500,
  "save_to_tensorboard": True,
  "logdir": "/home/sdevgupta/mine/OpenSeq2Seq/logs4",
  "max_grad_norm":1.,

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 10000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 20000,
    "min_lr": 1e-5,
  },
  "dtype": tf.float32,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
            'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": Tacotron2Encoder,
  "encoder_params": {
    "cnn_dropout_prob": 0.5,
    "rnn_dropout_prob": 0.,
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
    "use_cudnn_rnn": True,
    "rnn_type": tf.contrib.cudnn_rnn.CudnnLSTM,
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
        }
      ],
      "num_rnn_layers": 1,
      "rnn_cell_dim": 128,
      "rnn_unidirectional": True,
      "rnn_type": tf.nn.rnn_cell.GRUCell,
      "emb_size": 512,
      'attention_layer_size': 512,
      "num_tokens": 32,
      "num_heads": 8,
    }
  },
  
  "loss": MainLoss,
  "loss_params": {
    "use_mask": False
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
	"mfcc_dims": 80,
    "dataset": dataset,
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    'dataset_location':dataset_location,
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min":data_min,
    "mel_type":'htk',
    "trim": trim,   
    "duration_max":1024,
    "duration_min":24,
    "exp_mag": exp_mag
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, train),
    ],
    "shuffle": True,
    "style_input": "wav"
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, val),
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
		os.path.join(dataset_location, "infer_different_book_samples.csv")
	],
    "duration_max":10000,
    "duration_min":0,
    "shuffle": False,
    "style_input": "wav"
  },
}
