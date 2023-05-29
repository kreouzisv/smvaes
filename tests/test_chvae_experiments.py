##################################################
###Experiments with  linear HVAE models in a non-centered form
##################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
#tf.config.list_physical_devices('GPU')
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from absl import flags
import os
import sys
from CHVAE_model import CHVAE
import tensorflow_datasets as tfds
from tqdm import tqdm
import pdb
from utils.utils import KID_score
import pandas as pd

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tfk.layers


#possible command line arguments
flags.DEFINE_string("sampler",
                     default="dsHMC",
                     help="MCMC sampler used")
flags.DEFINE_float("learning_rate",
                     default=0.001,
                     help="learning rate for VAE optimizers")
flags.DEFINE_float("learning_rate_mcmc",
                     default=0.001,
                     help="learning rate for MCMC adaptation")
flags.DEFINE_integer("num_leapfrog_steps",
                     default=1,
                     help="number of leapfrog steps")
flags.DEFINE_integer("num_MCMC_steps",
                     default=5,
                     help="number of MCMC steps")
flags.DEFINE_integer("epochs_vae",
                     default=500,
                     help="training epochs vae without mcmc")
flags.DEFINE_integer("epochs_mcmc",
                     default=500,
                     help="training epochs vae with mcmc")
flags.DEFINE_integer("id",
                     default=2,
                     help="id of run")
flags.DEFINE_list("latent_dims",
                     default=[50,100],
                     help="dimension of latent variables")
flags.DEFINE_string("likelihood",
                     default="Normal",
                     help="likelihood of the generator")
flags.DEFINE_bool("residual",
                  default = False,
                  help ="if residual parameterization is present in the hVAE model")
flags.DEFINE_bool("train_prior",
                  default = False,
                  help ="if residual parameterization is present in the hVAE model")

flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'simulation'),
    help="Directory to put the model's fit and outputs.")
flags.DEFINE_string(
    'data',
    default='simulation',
    help="data (small_mnist, simulation, omniglot).")
flags.DEFINE_bool(
    'diagonal_pre_cond',
    default=False,
    help="if pre-conditioning matrix is diagonal.")
flags.DEFINE_float("min_prior_scale",
                     default=0.0001,
                     help="min prior scale in prior distribution")
FLAGS = flags.FLAGS

#def main(argv):
#  del argv  # unused
if True:
  FLAGS(sys.argv)
  #save (command line) flags to file
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path=os.path.join(FLAGS.model_dir, FLAGS.data, str(FLAGS.latent_dims), str(FLAGS.id),
                    '{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}'.format(
                                      flags.FLAGS.num_MCMC_steps,
                                      flags.FLAGS.num_leapfrog_steps,
                                      flags.FLAGS.likelihood,
                                      flags.FLAGS.train_prior,
                                      flags.FLAGS.epochs_vae,
                                      flags.FLAGS.epochs_mcmc,
                                      flags.FLAGS.learning_rate,
                                      flags.FLAGS.learning_rate_mcmc,
                                      flags.FLAGS.residual,
                                      flags.FLAGS.sampler,
                                      flags.FLAGS.diagonal_pre_cond,
                                      flags.FLAGS.min_prior_scale
                                      ))
  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()

  #set seeds to id
  tf.random.set_seed(FLAGS.id)
  np.random.seed(FLAGS.id)
  

  latent_dims = FLAGS.latent_dims

  if FLAGS.data == 'simulation':
    #simulate data
    data_dim = 10
    train_size = 1000
    sigma_data = .5
    sigma_z = tf.ones([len(latent_dims)])#1./tf.range(1,1+len(latent_dims), dtype=tf.float32)
    W_L = tf.random.normal([data_dim, latent_dims[-1]])
    As = []
    W = []
    z = []
    z_mean = []

    for i in range(len(latent_dims)):
      eps_i = tf.random.normal([train_size, latent_dims[i]])
      if i == 0:
        z_mean_i = tf.zeros_like(eps_i)
        z_i = eps_i * sigma_z[i]
      else:
        A_i = (tf.random.normal([latent_dims[i], latent_dims[i - 1]]))
        z_i = tf.linalg.matvec(A_i, z_i) + eps_i * sigma_z[i]
        z_mean_i = tf.linalg.matvec(A_i, z_mean_i)
        As.append(A_i)
      z.append(z_i)
      z_mean.append(z_mean_i)
      if i < len(latent_dims)-1:
        W.append(tf.zeros([data_dim,latent_dims[i]]))
      else:
        W.append(W_L)
    z = tf.concat(z, -1)
    z_mean = tf.concat(z_mean, -1)
    W = tf.concat(W, -1)

    x = tf.linalg.matvec(W, z) + sigma_data * tf.random.normal([train_size, data_dim])
    x_mean = tf.linalg.matvec(W, z_mean)
    assert(len(latent_dims)==2)
    Lambda0 = tf.linalg.diag(sigma_z[0]**2 * tf.ones([latent_dims[0]]))


    z_cov = tf.concat(
      [tf.concat([Lambda0, tf.matmul(As[0], Lambda0)], 0),
       tf.concat([tf.matmul(Lambda0, As[0],  adjoint_b=True),
                  sigma_z[1]**2 * tf.ones([latent_dims[1]])+ tf.matmul(As[0], tf.matmul(Lambda0, As[0], adjoint_b=True))
                  ], 0)], 1)

    x_cov = tf.matmul(W, tf.matmul(z_cov, W, adjoint_b=True)) + sigma_data**2 * tf.eye(data_dim)
    x_llh_true = tf.reduce_mean(tfd.MultivariateNormalFullCovariance(loc = x_mean, covariance_matrix = x_cov).log_prob(x))

    batch_size = 1000
    data = (tf.data.Dataset.from_tensor_slices(x).shuffle(train_size).batch(batch_size))

  elif FLAGS.data in ['mnist','small_mnist']:
    #from https://colab.research.google.com/github/google-research/google-research/blob/master/linear_vae/DontBlameTheELBO.ipynb
    def load_mnist_data(train_take, test_take):
      '''Load a subset of the MNIST dataset, repeated.

      Args:
        take: The amount of training data to take.

      Returns:
        A shuffled subset of the MNIST training dataset.
      '''
      mnist_builder = tfds.builder("mnist")
      mnist_info = mnist_builder.info
      mnist_builder.download_and_prepare()
      datasets = mnist_builder.as_dataset()
      train_dataset = datasets['train']
      train_dataset = train_dataset.take(train_take).map(lambda x: tf.reshape(tf.cast(x['image'], tf.float32) / 255.0, [-1]))

      test_dataset = datasets['test']
      test_dataset = test_dataset.take(test_take).map(lambda x: tf.reshape(tf.cast(x['image'], tf.float32) / 255.0, [-1]))

      

      return train_dataset,test_dataset


    train_size = 2000 if FLAGS.data == 'small_mnist' else 50000
    test_size = 1000 if FLAGS.data == 'small_mnist' else 10000
    x, x_test = tfds.as_numpy(load_mnist_data(train_size,test_size))
    x = np.array(list(x))
    x_test = np.array(list(x_test))

    covariance = np.cov(x, rowvar = False)
    data_dim = x.shape[-1]
    batch_size = 200
    data = (tf.data.Dataset.from_tensor_slices(x).batch(batch_size))
    test_data = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))


  elif FLAGS.data in ['fashion_mnist']:
    #from https://colab.research.google.com/github/google-research/google-research/blob/master/linear_vae/DontBlameTheELBO.ipynb
    def load_fashion_mnist_data(train_take, test_take):
      '''Load a subset of the MNIST dataset, repeated.

      Args:
        take: The amount of training data to take.

      Returns:
        A shuffled subset of the MNIST training dataset.
      '''
      mnist_builder = tfds.builder("fashion_mnist")
      mnist_info = mnist_builder.info
      mnist_builder.download_and_prepare()
      datasets = mnist_builder.as_dataset()
      train_dataset = datasets['train']
      train_dataset = train_dataset.take(train_take).map(lambda x: tf.reshape(tf.cast(x['image'], tf.float32) / 255.0, [-1]))

      test_dataset = datasets['test']
      test_dataset = test_dataset.take(test_take).map(lambda x: tf.reshape(tf.cast(x['image'], tf.float32) / 255.0, [-1]))

      

      return train_dataset,test_dataset


    train_size = 2000 if FLAGS.data == 'small_mnist' else 50000
    test_size = 1000 if FLAGS.data == 'small_mnist' else 10000
    x, x_test = tfds.as_numpy(load_fashion_mnist_data(train_size,test_size))
    x = np.array(list(x))
    x_test = np.array(list(x_test))

    covariance = np.cov(x, rowvar = False)
    data_dim = x.shape[-1]
    batch_size = 200
    data = (tf.data.Dataset.from_tensor_slices(x).batch(batch_size))
    test_data = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))


  elif FLAGS.data in ['omniglot']:
    def load_omniglot_data(take):
      '''Load a subset of the omniglot dataset, repeated.

      Args:
        take: The amount of training data to take.

      Returns:
        A shuffled subset of the omniglot training dataset.
      '''
      omniglot_builder = tfds.builder("omniglot")
      omniglot_builder.download_and_prepare()
      datasets = omniglot_builder.as_dataset()
      train_dataset = datasets['train']
      test_dataset = datasets['test']

      #test_dataset = train_dataset.take(take).map(lambda x: tf.reshape(tf.cast(x['image'], tf.float32), [-1]))
      #train_dataset = train_dataset.take(take).map(lambda x: tf.reshape(tf.cast(x['image'], tf.float32), [-1]))
      return train_dataset


    train_size = 19280
    x = np.array(list(tfds.as_numpy(load_omniglot_data(train_size))))

    covariance = np.cov(x, rowvar = False)
    data_dim = x.shape[-1]
    batch_size = 200
    data = (tf.data.Dataset.from_tensor_slices(x).shuffle(train_size).batch(batch_size))


  ##############
  #Define VAE model
  ##############

  # class ConstantLayer(tf.keras.layers.Layer):
  #   def __init__(self, dims):
  #     super(ConstantLayer, self).__init__()
  #     self.constant = tf.Variable(tfd.Uniform(.005, .01).sample() * tf.ones(dims))
  #
  #   def call(self, inputs):
  #     return self.constant
  class ConstantLayer_(tf.keras.layers.Layer):
    def __init__(self, output_units, const_init=tf.zeros_initializer(), fixed_init=None):
      super(ConstantLayer_, self).__init__()
      self.output_units = output_units
      self.const_init = const_init
      self.fixed_init = fixed_init

    def build(self, input_shape):  # Create the state of the layer (weights)
      self.w = 0.*tf.Variable(
        initial_value = tf.random_normal_initializer()(shape = (self.output_units, input_shape[-1]),
                               dtype = 'float32'),
        trainable = False)
      if self.fixed_init is None:
        b_init = self.const_init
        self.b = 0.*tf.Variable(
          initial_value = b_init(shape = (self.output_units,), dtype = 'float32'),
          trainable = False)
      else:
        self.b = 0.*tf.Variable(
          initial_value = self.fixed_init,
          trainable = False)

    def call(self, inputs):  # Defines the computation from inputs to outputs
      #return tf.matmul(inputs, self.w_full) + self.b
      return tf.linalg.matvec(self.w, inputs) + self.b

  class ConstantLayer(tf.keras.layers.Layer):
    def __init__(self, output_units, const_init=tf.zeros_initializer(), fixed_init=None):
      super(ConstantLayer, self).__init__()
      self.output_units = output_units
      self.const_init = const_init
      self.fixed_init = fixed_init

    def build(self, input_shape):  # Create the state of the layer (weights)
      self.w = 0.*tf.Variable(
        initial_value = tf.random_normal_initializer()(shape = (self.output_units, input_shape[-1]),
                               dtype = 'float32'),
        trainable = False)
      if self.fixed_init is None:
        b_init = self.const_init
        self.b = tf.Variable(
          initial_value = b_init(shape = (self.output_units,), dtype = 'float32'),
          trainable = True)
      else:
        self.b = tf.Variable(
          initial_value = self.fixed_init,
          trainable = True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
      #return tf.matmul(inputs, self.w_full) + self.b
      return tf.linalg.matvec(self.w, inputs) + self.b


  class LinearDeterministicLayer(tf.keras.layers.Layer):
    def __init__(self, output_units, w_init = tf.random_normal_initializer(), zero_mean_bias = False):
      super(LinearDeterministicLayer, self).__init__()
      self.output_units = output_units
      self.w_init = w_init
      self.zero_mean_bias = zero_mean_bias

    def build(self, input_shape):  # Create the state of the layer (weights)
      self.w = tf.Variable(
        initial_value = self.w_init(shape = (self.output_units, input_shape[-1]),
                               dtype = 'float32'),
        trainable = True)
      # self.w = tf.Variable(
      #   initial_value = self.w_init(shape = (input_shape[-1], self.output_units),
      #                          dtype = 'float32'),
      #   trainable = True)
      self._input_shape = input_shape
      b_init = tf.zeros_initializer()
      if self.zero_mean_bias:
        self._b = tf.Variable(
          initial_value = b_init(shape = (self.output_units,), dtype = 'float32'),
          trainable = True)
        self.b = tf.concat([tf.zeros_like(self._b), self._b], 0)
      else:
        self.b = tf.Variable(
          initial_value = b_init(shape = (2 * self.output_units,), dtype = 'float32'),
          trainable = True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
      #return tf.matmul(inputs, self.w_full) + self.b
      w_full = tf.concat([self.w, tf.zeros([self.output_units, self._input_shape[-1]])], 0)
      return tf.linalg.matvec(w_full, inputs) + self.b



  if FLAGS.data == 'simulation_':

    deterministic_layers_dims = [2*l for l in latent_dims] #top-down-direction
    L = len(deterministic_layers_dims)
    decoder_layers = []
    for l in range(L):
      decoder = tf.keras.Sequential([
         tfk.layers.InputLayer(input_shape = (deterministic_layers_dims[l], )),
         tfk.layers.Lambda(lambda x: x)])
      decoder_layers.append(decoder)


    encoder_layers = []
    for l in range(L):
      encoder =tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape = ((2*deterministic_layers_dims[l]))),
          LinearDeterministicLayer(output_units = latent_dims[l])])
      encoder_layers.append(encoder)

    bottom_up_deterministic_layers = []
    for l in range(L):
      # Build deterministic block
      if l == 0:
        linear_layer = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape = (data_dim,)),
          tf.keras.layers.Dense(deterministic_layers_dims[-1])
        ])
      else:
        linear_layer = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape = (deterministic_layers_dims[L-(l)],)),
          tf.keras.layers.Dense(deterministic_layers_dims[L-(l+1)],
                                kernel_initializer = tfk.initializers.RandomNormal(stddev=.05))
        ])
      bottom_up_deterministic_layers.append(linear_layer)



    
    if FLAGS.residual == True:
      top_down_deterministic_layers = []
      for l in range(L):
        # Build deterministic block
        if l==0:
          linear_feature = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (1,)),
            ConstantLayer_(output_units = deterministic_layers_dims[0])])

        else:
          linear_feature = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[l-1] + deterministic_layers_dims[l-1])),
              # tf.keras.layers.Dense(2 * latent_dims[l])])
            LinearDeterministicLayer(output_units = latent_dims[l])])
            #tf.keras.layers.Dense(2*latent_dims[l],
            #                      kernel_initializer = tfk.initializers.RandomNormal(stddev=.05))])

          #linear_feature = tf.keras.layers.Concatenate([mean_feature, log_var_feature])

        top_down_deterministic_layers.append(linear_feature)

      observation_layer = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[-1] + deterministic_layers_dims[-1])),
            LinearDeterministicLayer(output_units = data_dim)])

    else:
      top_down_deterministic_layers = []
      for l in range(L):
        # Build deterministic block
        if l==0:
          linear_feature = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (1,)),
            ConstantLayer_(output_units = deterministic_layers_dims[0])])

        else:
          linear_feature = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[l-1],) if l>0 else (1,)),
              # tf.keras.layers.Dense(2 * latent_dims[l])])
            LinearDeterministicLayer(output_units = latent_dims[l])])
            #tf.keras.layers.Dense(2*latent_dims[l],
            #                      kernel_initializer = tfk.initializers.RandomNormal(stddev=.05))])

          #linear_feature = tf.keras.layers.Concatenate([mean_feature, log_var_feature])

        top_down_deterministic_layers.append(linear_feature)

      observation_layer = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[1],)),
            LinearDeterministicLayer(output_units = data_dim)])

  else:

    deterministic_layers_dims = [2*l for l in latent_dims]
    # deterministic_layers_dims = [l for l in latent_dims]
     #top-down-direction
    L = len(deterministic_layers_dims)
    decoder_layers = []
    for l in range(L):
      if l == 0:
        decoder = tf.keras.Sequential([
           tfk.layers.InputLayer(input_shape = (deterministic_layers_dims[l], )),
            # tf.keras.layers.Dense(200,'relu'),
            # tf.keras.layers.Dense(200,'relu'),
            # tf.keras.layers.Dense(2 * latent_dims[l])])
            # gauss case
            # tf.keras.layers.Dense(2*latent_dims[l])
            # linear case
            tfk.layers.Lambda(lambda x: x)])
      else:
        decoder = tf.keras.Sequential([
           tfk.layers.InputLayer(input_shape = (deterministic_layers_dims[l], )),
            tf.keras.layers.Dense(200,'relu'),
            tf.keras.layers.Dense(200,'relu'),
            tf.keras.layers.Dense(2 * latent_dims[l])])
      decoder_layers.append(decoder)


    encoder_layers = []
    for l in range(L):
      encoder =tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape = ((2*deterministic_layers_dims[l],))),
          tf.keras.layers.Dense(200,'relu'),
          tf.keras.layers.Dense(200,'relu'),
          tf.keras.layers.Dense(2 * latent_dims[l])])
          # LinearDeterministicLayer(output_units = latent_dims[l])])
      encoder_layers.append(encoder)



    bottom_up_deterministic_layers = []
    for l in range(L):
      # Build deterministic block
      if l == 0:
        linear_layer = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape = (data_dim,)),
          tf.keras.layers.Dense(200,'relu'),
          tf.keras.layers.Dense(200,'relu'),
          tf.keras.layers.Dense(deterministic_layers_dims[-1])
        ])
      else:
        linear_layer = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape = (deterministic_layers_dims[L-(l)],)),
          tf.keras.layers.Dense(200,'relu'),
          tf.keras.layers.Dense(200,'relu'),
          tf.keras.layers.Dense(deterministic_layers_dims[L-(l+1)],
                                kernel_initializer = tfk.initializers.RandomNormal(stddev=.05))
        ])
      bottom_up_deterministic_layers.append(linear_layer)

    if FLAGS.residual == True:
      top_down_deterministic_layers = []
      for l in range(L):
        # Build deterministic block
        if l==0:
          # always constant
          linear_feature = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (1,)),
            # tf.keras.layers.Dense(20,'relu'),
            # tf.keras.layers.Dense(20,'relu'),
            # tf.keras.layers.Dense(deterministic_layers_dims[0])])
            ConstantLayer_(output_units = deterministic_layers_dims[0])])

        else:
          linear_feature = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[l-1] + deterministic_layers_dims[l-1])),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(deterministic_layers_dims[l]),
            # LinearDeterministicLayer(output_units = latent_dims[l])
            ])
            #tf.keras.layers.Dense(2*latent_dims[l],
            #                      kernel_initializer = tfk.initializers.RandomNormal(stddev=.05))])

          #linear_feature = tf.keras.layers.Concatenate([mean_feature, log_var_feature])

        top_down_deterministic_layers.append(linear_feature)

    else:
      top_down_deterministic_layers = []
      for l in range(L):
        # Build deterministic block
        if l==0:
          # always constant
          linear_feature = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (1,)),
            # tf.keras.layers.Dense(20,'relu'),
            # tf.keras.layers.Dense(20,'relu'),
            # tf.keras.layers.Dense(deterministic_layers_dims[0])])
            ConstantLayer_(output_units = deterministic_layers_dims[0])])

        else:
          linear_feature = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[l-1],) if l>0 else (1,)),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(deterministic_layers_dims[l]),
            # LinearDeterministicLayer(output_units = latent_dims[l])
            ])
            #tf.keras.layers.Dense(2*latent_dims[l],
            #                      kernel_initializer = tfk.initializers.RandomNormal(stddev=.05))])

          #linear_feature = tf.keras.layers.Concatenate([mean_feature, log_var_feature])

        top_down_deterministic_layers.append(linear_feature)


    # observation_layer = tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape = (latent_dims[1],)),
    #         tf.keras.layers.Dense(200,'relu'),
    #         tf.keras.layers.Dense(200,'relu'),
    #         tf.keras.layers.Dense(data_dim + data_dim),
    #       # LinearDeterministicLayer(output_units = data_dim),
    #         tf.keras.layers.Activation('sigmoid')
    #       ])

    if FLAGS.residual == True:
      observation_layer = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[-1] + deterministic_layers_dims[-1])),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(data_dim),
            # LinearDeterministicLayer(output_units = data_dim),
              # tf.keras.layers.Activation('sigmoid')
            ])

    else:
      observation_layer = tf.keras.Sequential([
              tf.keras.layers.InputLayer(input_shape = (latent_dims[-1])),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(200,'relu'),
              tf.keras.layers.Dense(data_dim),
            # LinearDeterministicLayer(output_units = data_dim),
              # tf.keras.layers.Activation('sigmoid')
            ])

    # observation_layer = tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape = (latent_dims[-1],)),
    #         LinearDeterministicLayer(output_units = data_dim),
    #       ])

    # one layer case
    # observation_layer = tf.keras.Sequential([
    #         tf.keras.layers.InputLayer(input_shape = (latent_dims[-1],)),
    #         LinearDeterministicLayer(output_units = data_dim // 2),
    #       ])

  #######
  #Diagonal or full cholesky matrix for preconditioning
  #######

  if FLAGS.diagonal_pre_cond:
    diag_pre_cond_nn = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = (np.product(data_dim),)),
      tf.keras.layers.Flatten(),
      ConstantLayer(np.sum(latent_dims), const_init = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.05))])

    def pre_cond_fn(x):
      return tf.linalg.LinearOperatorDiag(diag = diag_pre_cond_nn(x))

    pre_cond_params = diag_pre_cond_nn.trainable_variables
    diag_pre_cond_nn.summary()


  else:
    init_chol = .01 * tfd.CholeskyLKJ(dimension=sum(latent_dims), concentration=2.).sample()
    chol_pre_cond_nn = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = data_dim),
      tf.keras.layers.Flatten(),
      ConstantLayer(output_units = np.sum(FLAGS.latent_dims)**2,
                    fixed_init = tf.reshape(init_chol,[sum(latent_dims)**2])),
      tf.keras.layers.Reshape([np.sum(np.sum(FLAGS.latent_dims)),np.sum(FLAGS.latent_dims)])])

    #ConstantLayer([np.sum(FLAGS.latent_dims), np.sum(FLAGS.latent_dims)])])

    chol_pre_cond_nn.summary()
    def pre_cond_fn(x):
      return tf.linalg.LinearOperatorLowerTriangular(
        tril = (chol_pre_cond_nn(x))
      )
    pre_cond_params = chol_pre_cond_nn.trainable_variables





  def generate_and_save_images(model_, epoch, test_sample):
    if FLAGS.data != 'omniglot':
      image_shape = int(np.sqrt(test_sample.shape[-1]))
    else:
      image_shape = int(np.sqrt(test_sample.shape[-1])/3)

    predictions = model_.mean_reconstruced_image(tf.reshape(test_sample, [test_sample.shape[0], -1]))

    if FLAGS.data != 'omniglot':
      predictions = tf.reshape(predictions, [test_sample.shape[0], image_shape, image_shape])

    else:
      predictions = tf.reshape(predictions, [test_sample.shape[0], image_shape, image_shape, 3])



    fig = plt.figure(figsize = (4, 4))
    for i in range(16):
      plt.subplot(4, 4, i + 1)
      if FLAGS.data != 'omniglot':
        plt.imshow(predictions[i, :, :], cmap = 'gray')
      else:
        plt.imshow(predictions[i, :, :, :], cmap = 'rgp')
      plt.axis('off')
    plt.savefig(os.path.join(path,'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()


  def generate_prior_predictions(model_, epoch, batch_size = 200):
    image_shape = int(np.sqrt(data_dim))
    predictions = model_.mean_prior_image(batch_size)

    predictions = tf.reshape(predictions, [batch_size, image_shape, image_shape])
    fig = plt.figure(figsize = (8, 10))
    for i in range(80):
      plt.subplot(10, 8, i + 1)
      plt.imshow(predictions[i, :, :], cmap = 'gray')
      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(os.path.join(path,'prior_generated_image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()






  model_0 = CHVAE(
    encoder_layers = encoder_layers,
    decoder_layers = decoder_layers,
    top_down_deterministic_layers = top_down_deterministic_layers,
    bottom_up_deterministic_layers = bottom_up_deterministic_layers,
    input_dim = np.prod(data_dim),
    latent_dims = FLAGS.latent_dims,
    generative_deterministic_dims = deterministic_layers_dims,
    pre_cond_fn = pre_cond_fn,
    pre_cond_params = pre_cond_params,
    likelihood = FLAGS.likelihood,
    sampler = FLAGS.sampler,
    num_MCMC_steps = 0,
    num_leapfrog_steps = FLAGS.num_leapfrog_steps,
    learning_rate = FLAGS.learning_rate,
    learning_rate_mcmc = FLAGS.learning_rate_mcmc,
    observation_layer = observation_layer,
    use_residual = FLAGS.residual
    )

  #########
  #Train the model
  #########
  encoder_losses = []
  decoder_losses = []
  decoder_losses_init = []

  for epoch in tqdm(range(0, FLAGS.epochs_vae + 1),desc = 'Training'):
    if FLAGS.data == 'simulation_':
      # model_0.train_step(tf.reshape(x, [x.shape[0], -1]))
      for x_batch in data:
        model_0.train_step(tf.reshape(x_batch, [x_batch.shape[0], -1]))
  

    elif FLAGS.data in ['mnist', 'small_mnist', 'omniglot','fashion_mnist']:
      for x_batch in data:
        model_0.train_step(tf.reshape(x_batch, [x_batch.shape[0], -1]))

      loss_encoder = tf.keras.metrics.Mean()
      loss_decoder = tf.keras.metrics.Mean()
      loss_decoder_init = tf.keras.metrics.Mean()

    if FLAGS.data in ['mnist', 'small_mnist', 'omniglot','fashion_mnist']:
      if epoch % 10 == 0:
        for test_x in test_data:
          encoder_loss_, decoder_loss_, decoder_loss_init_ = model_0.compute_loss(tf.reshape(x_batch, [x_batch.shape[0], -1]))
          loss_encoder(encoder_loss_)
          loss_decoder(decoder_loss_)
          loss_decoder_init(decoder_loss_init_)

        generate_and_save_images(model_0, epoch, x_batch)
        generate_prior_predictions(model_0, epoch)
        # samples = test_size
        # real_images = tf.reshape(x_test[:samples,::], (samples, 28,28,1))
        # synthetic_data= model_0.synthesize_data(samples = samples)
        # kid = KID_score(real_images, synthetic_data,samples = samples)
        # print('kid_score', kid)
        

        encoder_losses.append(loss_encoder.result())
        decoder_losses.append(loss_decoder.result())
        decoder_losses_init.append(loss_decoder_init.result())

        print(
            'Epoch: {}, encoder_loss: {},decoder_loss: {}, decoder_loss_init: {}'
              .format(epoch, loss_encoder.result(), loss_decoder.result(), loss_decoder_init.result()))

  if FLAGS.data in ['mnist', 'small_mnist', 'omniglot','fashion_mnist']:
    samples = test_size
    real_images = tf.reshape(x_test[:samples,::], (samples, 28,28,1))
    synthetic_data= model_0.synthesize_data(samples = samples)
    kid_vae = KID_score(real_images, synthetic_data,samples = samples)
    print('kid_score_VAE', kid_vae)

    pd.DataFrame(['KID score vae', ': ', kid_vae]).to_csv(os.path.join(path,'KID score vae model'),index=False)



  if FLAGS.data == 'simulation_':
    min_prior_scale = FLAGS.min_prior_scale #this is always added to the scale parameters in the VAE model
    A1_learned_0 = top_down_deterministic_layers[1].trainable_weights[0]
    sigma_z0_learned_0 = min_prior_scale + tf.math.exp(.5 * 0.)
    sigma_z1_learned_0 = min_prior_scale + tf.math.exp(.5 * top_down_deterministic_layers[1].trainable_weights[1][latent_dims[1]:])



    # A_z_eps = tf.concat([
    #   tf.concat([tf.linalg.diag(sigma_z0_learned_0**2 * tf.ones([latent_dims[0]])),tf.matmul(A1_learned_0, tf.linalg.diag(sigma_z0_learned_0**2 * tf.ones([latent_dims[0]])))], 0),
    #   tf.concat([tf.zeros([latent_dims[0], latent_dims[1]]), tf.linalg.diag(sigma_z1_learned_0**2 * tf.ones([latent_dims[1]]))], 0)], -1)
    if FLAGS.residual:
      pdb.set_trace()
      prior_z_cov_learned_0 = tf.concat([
      tf.concat([tf.linalg.diag(sigma_z0_learned_0**2 * tf.ones([latent_dims[0]])),tf.transpose(A1_learned_0)], 1),
      tf.concat([A1_learned_0, tf.matmul(A1_learned_0, A1_learned_0, adjoint_b = True) + tf.linalg.diag(sigma_z1_learned_0**2 * tf.ones([latent_dims[1]]))], 1)], 0)

    else:
      prior_z_cov_learned_0 = tf.concat([
        tf.concat([tf.linalg.diag(sigma_z0_learned_0**2 * tf.ones([latent_dims[0]])),tf.transpose(A1_learned_0)], 1),
        tf.concat([A1_learned_0, tf.matmul(A1_learned_0, A1_learned_0, adjoint_b = True) + tf.linalg.diag(sigma_z1_learned_0**2 * tf.ones([latent_dims[1]]))], 1)], 0)



    eigenvalues_z_learned_prior_0 = tf.linalg.eigh(tf.linalg.inv(prior_z_cov_learned_0))[0]
    condition_number_z_learned_prior_0 = np.max(abs(eigenvalues_z_learned_prior_0))/np.min(abs(eigenvalues_z_learned_prior_0))
    print('condition_number_z_learned_prior_0', condition_number_z_learned_prior_0)


    # print(prior_z_cov_learned_0)

    W_learned_0 = tf.concat([tf.zeros([data_dim,latent_dims[0]]), observation_layer.trainable_variables[0]], 1)
    sigma_data_learned_0 = 1e-4 + tf.math.exp(.5 * observation_layer.trainable_variables[1][data_dim:])

    cov_xz_learned_0 = tf.matmul(W_learned_0, prior_z_cov_learned_0)
    cov_x_learned_0 = tf.matmul(W_learned_0, tf.matmul(prior_z_cov_learned_0, W_learned_0, adjoint_b = True)) \
                      + sigma_data_learned_0 **2 * tf.eye(data_dim)



    # posterior_eps_cov_learned_0 = tf.eye(sum(latent_dims)) - tf.matmul(
    #   W_A_z_eps_learned_0, tf.matmul(tf.linalg.inv(cov_x_learned_0), W_A_z_eps_learned_0), adjoint_a = True)
    # tf.debugging.assert_equal(posterior_eps_cov_learned_0,
    #                           tf.eye(sum(latent_dims)) - tf.matmul(
    #                             W_A_z_eps, tf.matmul(tf.linalg.inv(
    #                               tf.matmul(W_A_z_eps, W_A_z_eps, adjoint_b=True) + sigma_data_learned_0 **2 * tf.eye(data_dim)
    #                             ), W_A_z_eps), adjoint_a = True)
    #                           )

    posterior_z_cov_learned_0 = prior_z_cov_learned_0 - tf.matmul(
      cov_xz_learned_0, tf.matmul(tf.linalg.inv(cov_x_learned_0), cov_xz_learned_0), adjoint_a = True)

    
    
    eigenvalues_z_learned_0 = tf.linalg.eigh(tf.linalg.inv(posterior_z_cov_learned_0))[0]
    condition_number_z_learned_0 = np.max(abs(eigenvalues_z_learned_0))/np.min(abs(eigenvalues_z_learned_0))
    # print('condition_number_eps_learned_0',condition_number_eps_learned_0)
    print('condition_number_z_learned_0',condition_number_z_learned_0)

    #marginal log-likelihood
    z0_mean_learned_0 = tf.zeros(FLAGS.latent_dims[0])
    z1_mean_learned_0 = tf.linalg.matvec(A1_learned_0, z0_mean_learned_0) + \
                      top_down_deterministic_layers[1].trainable_weights[1][:latent_dims[1]]
    z_mean_learned_0 = tf.concat([z0_mean_learned_0, z1_mean_learned_0], -1)


    x_mean_learned_0 = tf.linalg.matvec(W_learned_0, z_mean_learned_0) + observation_layer.trainable_weights[1][:data_dim]
    x_log_prob_learned_0 = tfd.MultivariateNormalFullCovariance(loc = x_mean_learned_0, covariance_matrix = cov_x_learned_0).log_prob(x)
    x_llh_learned_0 = tf.reduce_mean(x_log_prob_learned_0)
  

  model = CHVAE(
    encoder_layers = encoder_layers,
    decoder_layers = decoder_layers,
    top_down_deterministic_layers = top_down_deterministic_layers,
    bottom_up_deterministic_layers = bottom_up_deterministic_layers,
    input_dim = np.prod(data_dim),
    latent_dims = FLAGS.latent_dims,
    generative_deterministic_dims = deterministic_layers_dims,
    pre_cond_fn = pre_cond_fn,
    pre_cond_params = pre_cond_params,
    likelihood = FLAGS.likelihood,
    sampler = FLAGS.sampler,
    num_MCMC_steps = FLAGS.num_MCMC_steps,
    num_leapfrog_steps = FLAGS.num_leapfrog_steps,
    learning_rate = FLAGS.learning_rate,
    learning_rate_mcmc = FLAGS.learning_rate_mcmc,
    observation_layer = observation_layer,
    train_prior = FLAGS.train_prior,
    use_residual = FLAGS.residual
    )

  for epoch in tqdm(range(0, FLAGS.epochs_mcmc + 1), desc = 'Training'):
    if FLAGS.data == 'simulation_':
      # model_0.train_step(tf.reshape(x, [x.shape[0], -1]))
      for x_batch in data:
        model.train_step(tf.reshape(x_batch, [x_batch.shape[0], -1]))
  

    elif FLAGS.data in ['mnist', 'small_mnist', 'omniglot','fashion_mnist']:
      for x_batch in data:
        model.train_step(tf.reshape(x_batch, [x_batch.shape[0], -1]))

      loss_encoder = tf.keras.metrics.Mean()
      loss_decoder = tf.keras.metrics.Mean()
      loss_decoder_init = tf.keras.metrics.Mean()
    if FLAGS.data in ['mnist', 'small_mnist', 'omniglot','fashion_mnist']:
      if epoch % 10 == 0:
        for test_x in test_data:
          encoder_loss_, decoder_loss_, decoder_loss_init_ = model.compute_loss(tf.reshape(x_batch, [x_batch.shape[0], -1]))
          loss_encoder(encoder_loss_)
          loss_decoder(decoder_loss_)
          loss_decoder_init(decoder_loss_init_)
        

        encoder_losses.append(loss_encoder.result())
        decoder_losses.append(loss_decoder.result())
        decoder_losses_init.append(loss_decoder_init.result())
        generate_and_save_images(model, 10000+epoch, x_batch)
        generate_prior_predictions(model, 10000+epoch)

        print(
            'Epoch: {}, encoder_loss: {},decoder_loss: {}, decoder_loss_init: {}'
              .format(epoch, loss_encoder.result(), loss_decoder.result(), loss_decoder_init.result()))

  if FLAGS.data in ['mnist', 'small_mnist', 'omniglot','fashion_mnist']:
    samples = test_size
    real_images = tf.reshape(x_test[:samples,::], (samples, 28,28,1))
    synthetic_data= model.synthesize_data(samples = samples)
    kid_vae = KID_score(real_images, synthetic_data,samples = samples)
    print('kid_score_VAE', kid_vae)

    pd.DataFrame(['KID score mcmc', ': ', kid_vae]).to_csv(os.path.join(path,'KID score vae mcmc model'),index=False)

  # for epoch in tqdm(range(FLAGS.epochs_mcmc), desc = 'Training'):
  #   if FLAGS.data == 'simulation_':
  #     # model_0.train_step(tf.reshape(x, [x.shape[0], -1]))
  #     for x_batch in data:
  #       model.train_step(tf.reshape(x_batch, [x_batch.shape[0], -1]))
  #   elif FLAGS.data in ['mnist','small_mnist', 'omniglot']:
  #     for x_batch in data:
  #       model.train_step(tf.reshape(x_batch, [x_batch.shape[0], -1]))
  #     if epoch % 10 == 0:


  #       generate_and_save_images(model_0, 10000+epoch, x_batch)
  #       generate_prior_predictions(model_0, 10000+epoch)

  # samples = test_size
  # real_images = tf.reshape(x_test[:samples,::], (samples, 28,28,1))
  # synthetic_data= model.synthesize_data(samples = samples)
  # kid = KID_score(real_images, synthetic_data,samples = samples)
  # print('kid_score_mcmc', kid)
  # pd.DataFrame(['KID score mcmc', ': ', kid]).to_csv(os.path.join(path,'KID score model'),index=False)
  plt.clf()
  fig = plt.figure(figsize = (10, 10))
  plt.plot(encoder_losses, label = "encoder_loss")
  plt.plot(decoder_losses, label = "decoder_loss_MCMC")
  plt.plot(decoder_losses_init, label = "decoder_loss")
  #plt.plot(KL_losses, label = 'KL-Divergence')
  #plt.plot(speed_measure_losses, label = "speed_measure_loss")
  plt.legend()
  plt.savefig(os.path.join(path,'Loss'))




  if FLAGS.data == 'simulation_':
    A1_learned = top_down_deterministic_layers[1].trainable_weights[0]
    sigma_z0_learned = min_prior_scale + tf.math.exp(.5 * 0.)
    sigma_z1_learned = min_prior_scale + tf.math.exp(.5 * top_down_deterministic_layers[1].trainable_weights[1][latent_dims[1]:])

    prior_z_cov_learned = tf.concat([
      tf.concat([tf.linalg.diag(sigma_z0_learned**2 * tf.ones([latent_dims[0]])),tf.transpose(A1_learned)], 1),
      tf.concat([A1_learned, tf.matmul(A1_learned, A1_learned, adjoint_b = True)+ tf.linalg.diag(sigma_z1_learned**2 * tf.ones([latent_dims[1]]))], 1)], 0)

    eigenvalues_z_learned_prior = tf.linalg.eigh(tf.linalg.inv(prior_z_cov_learned))[0]
    condition_number_z_learned_prior = np.max(abs(eigenvalues_z_learned_prior))/np.min(abs(eigenvalues_z_learned_prior))
    print('condition_number_z_learned_prior', condition_number_z_learned_prior)

    W_learned = tf.concat([tf.zeros([data_dim,latent_dims[0]]), observation_layer.trainable_variables[0]], 1)
    sigma_data_learned = 1e-4 + tf.math.exp(.5 * observation_layer.trainable_variables[1][data_dim:])

    cov_xz_learned = tf.matmul(W_learned, prior_z_cov_learned)
    # Marcel why we had this with an identity matrix?
    cov_x_learned = tf.matmul(W_learned, tf.matmul(prior_z_cov_learned, W_learned, adjoint_b = True)) \
                      + sigma_data_learned **2 * tf.eye(data_dim)

    # cov_x_learned = tf.matmul(W_learned, tf.matmul(prior_z_cov_learned, W_learned, adjoint_b = True)) \
    #                   + cov_xz_learned

    # tf.debugging.assert_equal(posterior_eps_cov_learned_0,
    #                           tf.eye(sum(latent_dims)) - tf.matmul(
    #                             W_A_z_eps, tf.matmul(tf.linalg.inv(
    #                               tf.matmul(W_A_z_eps, W_A_z_eps, adjoint_b=True) + sigma_data_learned_0 **2 * tf.eye(data_dim)
    #                             ), W_A_z_eps), adjoint_a = True)
    #                           )

    posterior_z_cov_learned = prior_z_cov_learned - tf.matmul(cov_xz_learned, tf.matmul(tf.linalg.inv(cov_x_learned), cov_xz_learned), adjoint_a = True)

    eigenvalues_z_learned = tf.linalg.eigh(tf.linalg.inv(posterior_z_cov_learned))[0]
    condition_number_z_learned = np.max(abs(eigenvalues_z_learned))/np.min(abs(eigenvalues_z_learned))
    # print('condition_number_eps_learned',condition_number_eps_learned)
    print('condition_number_z_learned',condition_number_z_learned)

    
    # print(C)
    C = pre_cond_fn(x).to_dense()[0]
    transformed_hessian = tf.matmul(C, tf.matmul(tf.linalg.inv(posterior_z_cov_learned), C), adjoint_a = True)
    # transformed_hessian = tf.linalg.inv(posterior_z_cov_learned)
    transformed_eigenvalues_learned = tf.linalg.eigh(transformed_hessian)[0]
    transformed_condition_number_learned = np.max(abs(transformed_eigenvalues_learned))/np.min(abs(transformed_eigenvalues_learned))
    print('transformed_condition_number_learned',transformed_condition_number_learned)


    #marginal log-likelihood
    # z0_mean_learned = top_down_deterministic_layers[0].trainable_weights[0][:latent_dims[0]]
    z0_mean_learned = tf.zeros(FLAGS.latent_dims[0])
    z1_mean_learned = tf.linalg.matvec(A1_learned, z0_mean_learned) + \
                      top_down_deterministic_layers[1].trainable_weights[1][:latent_dims[1]]
    z_mean_learned = tf.concat([z0_mean_learned, z1_mean_learned], -1)


    x_mean_learned = tf.linalg.matvec(W_learned, z_mean_learned) + observation_layer.trainable_weights[1][:data_dim]
    x_log_prob_learned = tfd.MultivariateNormalFullCovariance(loc = x_mean_learned, covariance_matrix = cov_x_learned).log_prob(x)
    x_llh_learned = tf.reduce_mean(x_log_prob_learned)
    print('x_llh_true',x_llh_true)
    print('x_llh_learned_0',x_llh_learned_0)
    print('x_llh_learned',x_llh_learned)

    pd.DataFrame(['x_llh_true', ': ', x_llh_true,
                'x_llh_learned_0', ': ', x_llh_learned_0,
                'x_llh_learned', ': ', x_llh_learned,
                'transformed_condition_number_learned', ': ', transformed_condition_number_learned,
                'condition_number_z_learned', ': ', condition_number_z_learned,
                'condition_number_z_learned_prior', ': ', condition_number_z_learned_prior,
                'condition_number_z_learned_0', ': ', condition_number_z_learned_0,
                'condition_number_z_learned_prior_0', ': ', condition_number_z_learned_0,]).to_csv(os.path.join(path,'Simulation Results'),index=False)



    #check covariance of prior and posterior
    eps_samples_prior_learned, _, z_samples_prior_learned, _ = model.sample_prior(5000)
    prior_cov_learned_samples = tfp.stats.covariance(z_samples_prior_learned)
        