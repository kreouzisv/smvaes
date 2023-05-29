##################################################
###Experiments with a VAE models
##################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
#tf.config.list_physical_devices('GPU')
import tensorflow_probability as tfp
import numpy as np
import time
import matplotlib.pyplot as plt
from absl import app
from absl import flags
import sys
from VAE_model import VAE
import pandas as pd
from load_oasis_3 import load_oasis
from utils import generate_and_save_images
from utils import Callback_EarlyStopping
from utils import compute_mixture
#from utils import discretized_mix_logistic_loss
#from utils import sample_from_discretized_mix_logistic
from flow_based_generative_model import Flow_Generator
from utils import make_bijector_kwargs
from utils import sample_from_discretized_mix_logistic
from utils import predict_from_discretized_mix_logistic
from utils import resnet50_encoder, resnet50_decoder
#from utils import LogistixMixture
import seaborn as sns
from utils import FID_score, KID_score
from utils import preprocess_binary_images, preprocess_images, scale_images, preprocess_images_logistic_mixture, preprocess_images_log_normal
from utils import preprocess_images_logistic_mixture
from load_reduced_dataset import load_reduced_dataset
from sklearn.model_selection import train_test_split
from load_celebA import load_celebA
from tqdm import tqdm
from tqdm import trange
import math
import tensorflow_datasets as tfds
from load_adni import load_adni
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from extra_keras_datasets import emnist, svhn

from utils import PInputsData, PInputsGenerated, PseudoInputs
from scipy.io import loadmat


tfd = tfp.distributions
tfb = tfp.bijectors



#possible command line arguments
flags.DEFINE_string("sampler",
                     default="",
                     help="MCMC sampler used: gradHMC or gradMALA or dsNUTS or dsMALA or dsHMC or HMC")

flags.DEFINE_float("learning_rate",
                     default=0.001,
                     help="learning rate for all vae optimizers")

flags.DEFINE_float("learning_rate_mcmc",
                     default=0.01,
                     help="learning rate for all mcmc optimizers")

flags.DEFINE_integer("latent_dim",
                     default=20,
                     help="dimension of latent variables")

flags.DEFINE_integer("num_MCMC_steps",
                     default=0,
                     help="number of MCMC steps")

flags.DEFINE_integer("epochs",
                     default=2000,
                     help="training epochs")

flags.DEFINE_integer("sampling_init_epoch",
                     default=2001,
                     help="when to start mcmc")

flags.DEFINE_integer("id",
                     default=0,
                     help="id of run")

flags.DEFINE_integer("num_leapfrog_steps",
                     default=2,
                     help="if HMC is used specify gradient computations")

flags.DEFINE_bool("biased_grads",
                     default=True,
                     help="true if biased log accept grads are used")

flags.DEFINE_bool("eval_nll",
                     default=False,
                     help="if test nll is being computed during training")

flags.DEFINE_bool("eval_kid",
                     default=True,
                     help="if KID score is computed after training")

flags.DEFINE_bool("save_generated_data",
                     default=True,
                     help="store generated data")

flags.DEFINE_string("netw",
                     default="mlp",
                     help="network architecture")

flags.DEFINE_string("likelihood",
                     default="logistic",
                     help="Bernoulli or Normal or Log_Normal or logistic_mix or logistic")

flags.DEFINE_string("prior",
                     default="Isotropic_Gaussian",
                     help="Isotropic_Gaussian or Vamp_prior or IAF_prior or Real_NVP_prior or Gaussian_Mixture")

flags.DEFINE_integer("prior_mixtures",
                     default=50,
                     help="prior mixture or number of pseudo_inputs if GoM or Vamp being utilized")

flags.DEFINE_float("beta",
                     default=1.0,
                     help="KL anealing")

flags.DEFINE_float("obs_log_var",
                     default=-2.,
                     help="if trainable")

flags.DEFINE_integer("reduced_sample",
                     default=0,
                     help="dataset reduction if applicable")

flags.DEFINE_float("nll_var_scaling",
                     default=1.2,
                     help="scaling for nll proposal")

flags.DEFINE_integer("nll_particles",
                     default=1,
                     help="number of importance samples")

flags.DEFINE_string("oasis_class",
                     default="AD",
                     help="C or AD")

flags.DEFINE_string("oasis_transform",
                     default='',
                     help="None or rHC or lHC")

flags.DEFINE_integer("oasis_slice_id",
                     default=0,
                     help="ranges from 0-")

flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'adni_hope'),
    help="Directory to put the model's fit and outputs.")
flags.DEFINE_string(
    'data_set',
    default='adni',
    help="data set mnist or fashion_mnist or oasis or cifar10 or reduced_mnist or reduced_cifar")
flags.DEFINE_bool(
    'diagonal_pre_cond',
    default=True,
    help="if pre-conditioning matrix is diagonal.")
FLAGS = flags.FLAGS

def main(argv):
  del argv  # unused

  #save (command line) flags to file
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path=os.path.join(FLAGS.model_dir,flags.FLAGS.oasis_transform,flags.FLAGS.oasis_class,str(flags.FLAGS.oasis_slice_id),
                    '{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}'.format(
                      flags.FLAGS.data_set,
                      flags.FLAGS.sampler,
                      flags.FLAGS.num_MCMC_steps,
                      flags.FLAGS.biased_grads, 
                      flags.FLAGS.learning_rate,
                      flags.FLAGS.learning_rate_mcmc,
                      flags.FLAGS.id,
                      flags.FLAGS.likelihood,
                      flags.FLAGS.prior,
                      flags.FLAGS.sampler,
                      flags.FLAGS.netw,
                      flags.FLAGS.num_leapfrog_steps,
                      flags.FLAGS.diagonal_pre_cond,
                      flags.FLAGS.latent_dim,
                      flags.FLAGS.reduced_sample,
                      flags.FLAGS.obs_log_var,
                      flags.FLAGS.beta,
                      flags.FLAGS.nll_var_scaling,
                      flags.FLAGS.nll_particles,
                      flags.FLAGS.epochs,
                      flags.FLAGS.sampling_init_epoch,
                      ))
  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()

  #set seeds to id
  tf.random.set_seed(FLAGS.id)
  np.random.seed(FLAGS.id)


  if FLAGS.data_set == 'oasis':
    train_images, test_images = load_oasis(label = FLAGS.oasis_class, transform =FLAGS.oasis_transform, slice_id = FLAGS.oasis_slice_id, random_state = FLAGS.id)
    
    #test_images = train_images
    # print('train images shape',train_images.shape)
    # print('train images shape',test_images.shape)
    train_images, test_images = np.squeeze(train_images), np.squeeze(test_images)
    normalize = False
    binarization = 'static'

  if FLAGS.data_set == 'adni':
    train_images, test_images = load_adni(label = FLAGS.oasis_class,random_state = FLAGS.id)

    print(train_images.shape)
    print(test_images.shape)
    
    #test_images = train_images
    # print('train images shape',train_images.shape)
    # print('train images shape',test_images.shape)
    train_images, test_images = np.squeeze(train_images), np.squeeze(test_images)
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    normalize = False
    binarization = 'static'

  if FLAGS.likelihood == 'Bernoulli':
    train_images = preprocess_binary_images(train_images, normalize = normalize,binarization = binarization)
    test_images = preprocess_binary_images(test_images, normalize = normalize,binarization = binarization)
  train_size = train_images.shape[0]
  test_size = test_images.shape[0]

  batch_size = 256
  batch_size_test = 100



  train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                   .shuffle(train_size).batch(batch_size))

  test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                  .shuffle(test_size).batch(batch_size_test))

  data_dim = train_images.shape[1], train_images.shape[2], train_images.shape[3]


  ##############
  #Define VAE model
  ##############
  

  latent_dim = FLAGS.latent_dim

  if FLAGS.netw == 'mlp':
    encoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = data_dim),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation = "relu"),
        tf.keras.layers.Dense(200, activation = "relu"),
        # No activation
        tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )


    decoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
        tf.keras.layers.Dense(200, activation = "relu"),
        tf.keras.layers.Dense(200, activation = "relu"),
        # tf.keras.layers.Dense(np.prod(data_dim)),
        tf.keras.layers.Dense(np.prod(data_dim), activation = 'sigmoid'),
        tf.keras.layers.Reshape((data_dim[0],data_dim[1],data_dim[2]))
      ]
    )
  if FLAGS.netw == 'cnn_':

    # mnist architecture
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2), activation='relu'),
            #tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2, 2), activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Activation('relu'),
            tf.keras.layers.Flatten(),
            # sometimes use intermediate dim
            #tf.keras.layers.Dense(units=128),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*64),
            tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same',activation = 'linear'),
        ]
    )


    # cifar architecture
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )


    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units = data_dim[0]//8 * data_dim[1]//8 * 128, activation = 'linear'),
            tf.keras.layers.Reshape(target_shape = (data_dim[0]//8, data_dim[1]//8, 128)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same'),
        ]
    )

    



  if FLAGS.netw == 'cnn':

    encoder = tf.keras.Sequential(
      [ 
        
        tf.keras.layers.InputLayer(input_shape = data_dim),
        tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 256, kernel_size = (3,3),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 1024, kernel_size = (3,3),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )


    decoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
        tf.keras.layers.Dense(units = data_dim[0]//8 * data_dim[1]//8 * 512, activation = 'linear'),
        tf.keras.layers.Reshape(target_shape = (data_dim[0]//8, data_dim[1]//8, 512)),
        tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3,3), strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3,3),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3,3), strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size =  (3,3), strides = 1,padding='same'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Activation('sigmoid')


      ])

  if FLAGS.netw == 'cnn_oasis':

    encoder = tf.keras.Sequential(
      [ 
        
        tf.keras.layers.InputLayer(input_shape = data_dim),
        tf.keras.layers.Conv2D(filters = 64, kernel_size = (5, 5),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 128, kernel_size = (5, 5),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 256, kernel_size = (5, 5),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 512, kernel_size = (5, 5),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 1024, kernel_size = (5, 5),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )


    decoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
        tf.keras.layers.Dense(units = data_dim[0]//6 * data_dim[1]//6 * 512, activation = 'linear'),
        tf.keras.layers.Reshape(target_shape = (data_dim[0]//6, data_dim[1]//6, 512)),
        tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (5, 5), strides = 2,padding='same'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (5, 5), strides = 2,padding='same'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (5, 5), strides = 2,padding='same'),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size =  (5, 5), strides = 1,padding='same'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Activation('relu'),
        #tf.keras.layers.Activation('sigmoid')


      ])









  if FLAGS.netw == 'resnet':

    encoder = resnet50_encoder(input_shape =data_dim,latent_dim = FLAGS.latent_dim)
    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=data_dim[0]//8 * data_dim[1]//8 * 256),
            tf.keras.layers.Reshape(target_shape=(data_dim[0]//8, data_dim[1]//8, 256)),
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding='same',activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same',activation = 'linear'),
        ]
    )
    # decoder = tf.keras.Sequential(
    #   [
    #     tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
    #     tf.keras.layers.Dense(units = data_dim[0]//8 * data_dim[1]//8 * 256, activation = 'linear'),
    #     tf.keras.layers.Reshape(target_shape = (data_dim[0]//8, data_dim[1]//8, 256)),
    #     tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = 2),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation('relu'),
    #     tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = 2),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation('relu'),
    #     tf.keras.layers.Conv2DTranspose(filters = 1, kernel_size =  (3, 3), strides = 1),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Activation('linear')


    #   ])


  #######
  #Diagonal or full cholesky matrix for preconditioning
  #######
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


  if FLAGS.diagonal_pre_cond==True:

    diag_pre_cond_nn = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = (data_dim)),
      tf.keras.layers.Flatten(),
      ConstantLayer(latent_dim, const_init = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.05))])

    def pre_cond_fn(x):
      return tf.linalg.LinearOperatorDiag(diag = diag_pre_cond_nn(x))

    pre_cond_params = diag_pre_cond_nn.trainable_variables
    diag_pre_cond_nn.summary()

    # class ConstantLayer(tf.keras.layers.Layer):
    #   def __init__(self):
    #     super(ConstantLayer, self).__init__()
    #     #self.constant = tf.Variable(tfd.Normal(loc=0., scale = 0.1).sample(int(latent_dim* (latent_dim+1)/2)))
    #     self.constant=tf.Variable(0.001 * tf.ones([latent_dim]))
                                              
    #   def call(self, inputs):
    #     return self.constant


    # diag_pre_cond_nn = tf.keras.Sequential([
    #   tf.keras.layers.InputLayer(input_shape = data_dim),
    #   tf.keras.layers.Flatten(),
    #   ConstantLayer()])

    # def pre_cond_fn(x):
    #   return tf.linalg.LinearOperatorDiag(diag = diag_pre_cond_nn(x))

    # pre_cond_params = diag_pre_cond_nn.trainable_variables

  if FLAGS.diagonal_pre_cond==False:

  

    class ConstantLayer(tf.keras.layers.Layer):
      def __init__(self):
        super(ConstantLayer, self).__init__()
        cor = tfp.distributions.CholeskyLKJ(dimension = latent_dim, concentration = 2).sample()
        self.constant = tf.Variable(tfd.Uniform(.05, .1).sample() * cor )
      def call(self, inputs):
        return self.constant

    chol_pre_cond_nn = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = data_dim),
      tf.keras.layers.Flatten(),
      ConstantLayer()])

    chol_pre_cond_nn.summary()
    def pre_cond_fn(x):
      return tf.linalg.LinearOperatorLowerTriangular(
        tril = (chol_pre_cond_nn(x))
      )
    pre_cond_params = chol_pre_cond_nn.trainable_variables


  #pseudo_inputs = None

  #if FLAGS.prior == 'Vamp_prior':

    


  if FLAGS.netw == 'IAF':

    model = Flow_Generator(latent_dim , bijector,decoder,
      learning_rate = FLAGS.learning_rate, likelihood = FLAGS.likelihood, prior = FLAGS.prior, encoder = encoder)





  else:

    model_0 = VAE(data_dim,latent_dim, encoder, decoder, pre_cond_fn, pre_cond_params,
                             sampler = FLAGS.sampler, learning_rate = FLAGS.learning_rate,learning_rate_mcmc = FLAGS.learning_rate_mcmc,
                             num_MCMC_steps = FLAGS.num_MCMC_steps,
                             likelihood = FLAGS.likelihood, biased_grads = FLAGS.biased_grads,
                            prior = FLAGS.prior, pseudo_inputs = PInputsGenerated(data_dim),
                            num_leapfrog_steps = FLAGS.num_leapfrog_steps, beta = FLAGS.beta, obs_log_var =FLAGS.obs_log_var)

  model_0.encoder.summary()
  model_0.decoder.summary()



  #########
  #Train VAE without mcmc steps
  #########
  epochs = FLAGS.epochs
  num_examples_to_generate = 16

  # Pick a sample of the test set for generating output images
  assert batch_size >= num_examples_to_generate
  for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]



  test_result_interval = 100

  # Train, output loss and reconstructed images
  #generate_and_save_images(model, 0, test_sample)
  #generate_prior_images(model, 0)
  def generate_and_save_images(model, epoch, test_sample):

    if FLAGS.netw == 'IAF':

      batch_size = test_sample.shape[0]

      mean_z, logvar_z = tf.split(model.encoder(test_sample), num_or_size_splits = 2, axis = 1)

      eps = tf.random.normal(shape = mean_z.shape)

      init_state = eps * tf.exp(logvar_z * .5) + mean_z

      transformed_distribution = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=mean_z, scale_diag=tf.math.exp(logvar_z)),
      bijector=tfb.Invert(model.bijector))

      bijector_args = make_bijector_kwargs(model.bijector, {'made.': {'conditional_input': tf.reshape(test_sample, [batch_size,-1])}})

      z = transformed_distribution.sample(
      bijector_kwargs=bijector_args)



    else:

      mean_z, logvar_z = model.encode(test_sample)
      z, kernel_results = model.reparameterize(
        mean_z, logvar_z,
        target_log_prob_fn = model.target_log_prob_fn(test_sample),
        x = test_sample)



    if FLAGS.likelihood == 'logistic_mix':
      params = model.decode(z)
      predictions =  sample_from_discretized_mix_logistic(l= params, nr_mix = 10)

    elif FLAGS.likelihood == 'Normal' or FLAGS.likelihood == 'logistic':
      predictions,__ = model.decode(z)

    elif FLAGS.likelihood == 'Bernoulli':
      predictions = model.decode(z, True)
      #predictions = tfd.Bernoulli(probs = predictions).sample()

      #predictions = tfd.Normal(loc = mean_x, scale = tf.math.exp(log_var_x)).sample()
    elif FLAGS.likelihood == 'Categorical':

      predictions = model.decode(z) 
      #predictions = tf.nn.softmax(predictions)
      predictions = tfd.Categorical(logits = predictions).sample()
      predictions  = tf.expand_dims(predictions, axis = -1)


    fig = plt.figure(figsize = (4, 4))
    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i + 1)
      #plt.imshow(predictions[i, :, :, :])
      plt.imshow(predictions[i, :, :, :], cmap = 'gray')
      #plt.imshow(tf.cast(predictions[i, :, :, :], tf.uint8))
      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(os.path.join(path,'image_at_epoch_{:04d}.png'.format(epoch)))

  def generate_prior_images(model, epoch):

    prior_gen_mean = model.synthesize_data(samples = 200,mode = 'mean')
    #prior_gen_mode = model.synthesize_data(samples = 200,mode = 'mode')
    #prior_gen_sample = model.synthesize_data(samples = 200,mode = 'random_sample')



    fig = plt.figure(figsize = (8, 10))
    for i in range(80):
      plt.subplot(10, 8, i + 1)
      #plt.imshow(prior_gen_mean[i, :, :, :])
      plt.imshow(prior_gen_mean[i, :, :, :], cmap = 'gray')

      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(os.path.join(path,'prior_generated_mean_image_at_epoch_{:04d}.png'.format(epoch)))
    plt.clf()

  encoder_losses = []
  decoder_losses = []
  decoder_losses_init = []
  KL_losses = []

  train_marginal_likelihood_estimates = []
  test_marginal_likelihood_estimates = []
  test_marginal_likelihood_estimate_sd = []

  marginal_log_likelihood_estimates_1 = []
  marginal_log_likelihood_estimates_2 = []
  marginal_log_likelihood_estimates_3 = []
  ess_values_1 = []
  ess_values_2 = []
  ess_values_3 = []

  sampler_params = []
  grads_sampler = []
  speed_measure_losses = []
  sampler_acceptance_rates = []
  entropy_weights = []
  kids = []

  


  for epoch in tqdm(range(0, epochs + 1),desc = 'Training'):
    start_time = time.time()

    #beta = min(epoch/100, 1.)
    beta = 1.
    

    for train_x in train_dataset:
      if epoch >= FLAGS.sampling_init_epoch:
        model_0.train_step(train_x, mcmc = True, beta = beta)
      else:
        model_0.train_step(train_x, beta = beta)



    loss_encoder = tf.keras.metrics.Mean()
    loss_decoder = tf.keras.metrics.Mean()
    loss_decoder_init = tf.keras.metrics.Mean()
    loss_llh = tf.keras.metrics.Mean()
    bits_per_dim_ = tf.keras.metrics.Mean()
    loss_speed_measure = tf.keras.metrics.Mean()
    acceptance_rate = tf.keras.metrics.Mean()
    entropy_weight = tf.keras.metrics.Mean()

    loss_llh_1 = tf.keras.metrics.Mean()
    loss_llh_2 = tf.keras.metrics.Mean()
    loss_llh_3 = tf.keras.metrics.Mean()

    ess_1_ = tf.keras.metrics.Mean()
    ess_2_ = tf.keras.metrics.Mean()
    ess_3_ = tf.keras.metrics.Mean()






    loss_KL = tf.keras.metrics.Mean()




    if not epoch % test_result_interval == 0:
      continue

    for test_x in tqdm(test_dataset,'Testing'):
      # losses
      if epoch >= FLAGS.sampling_init_epoch:
        encoder_loss_,decoder_loss_,decoder_loss_init_,KL_loss = model_0.compute_loss(test_x,mcmc=True, beta = beta)

        if FLAGS.eval_nll == True:
          #marginal_log_likelihood_estimate_ = model_0.annealed_importance_sampling_estimate(x= test_x)
          marginal_log_likelihood_estimate_ = model_0.marginal_log_likelihood_estimate(x_test = test_x, num_particles = FLAGS.nll_particles, scaling = FLAGS.nll_var_scaling,mcmc=True)


      else:
        encoder_loss_,decoder_loss_,decoder_loss_init_,KL_loss = model_0.compute_loss(test_x,mcmc=False, beta = beta)
        if FLAGS.eval_nll == True:
          marginal_log_likelihood_estimate_ = model_0.marginal_log_likelihood_estimate(x_test = test_x, num_particles = FLAGS.nll_particles, scaling = FLAGS.nll_var_scaling,mcmc=False)
      if FLAGS.eval_nll == False:
        marginal_log_likelihood_estimate_ = [0,0]

      loss_encoder(encoder_loss_)
      loss_decoder(decoder_loss_)
      loss_decoder_init(decoder_loss_init_)
      loss_KL(KL_loss)

      loss_llh_1(marginal_log_likelihood_estimate_[0])
      ess_1_(marginal_log_likelihood_estimate_[1])
      # loss_llh_3(marginal_log_likelihood_estimate_[2])
      # ess_1_(marginal_log_likelihood_estimate_[3])
      # ess_2_(marginal_log_likelihood_estimate_[4])
      # ess_3_(marginal_log_likelihood_estimate_[5])

      
        

    end_time = time.time()

    encoder_losses.append(loss_encoder.result())
    decoder_losses.append(loss_decoder.result())
    decoder_losses_init.append(loss_decoder_init.result())

    marginal_log_likelihood_estimates_1.append(loss_llh_1.result())
    # marginal_log_likelihood_estimates_2.append(loss_llh_2.result())
    # marginal_log_likelihood_estimates_3.append(loss_llh_3.result())
    ess_values_1.append(ess_1_.result())
    # ess_values_2.append(ess_2_.result())
    # ess_values_3.append(ess_3_.result())

    KL_losses.append(loss_KL.result())
   

    if epoch % test_result_interval == 0:
      generate_and_save_images(model_0, epoch, test_sample)
      generate_prior_images(model_0, epoch)
      # if FLAGS.eval_kid == True:
      #   samples = train_images.shape[0]
      #   synthetic_data= np.array(model_0.synthesize_data(samples = samples))
      #   kid = KID_score(train_images[:samples,::], synthetic_data,samples = samples)
      #   kids.append(kid)
      # else:
      kid = 0

    print(
      'Epoch: {}, encoder_loss: {},decoder_loss: {}, decoder_loss_init: {}, test_marginal_likelihood_estimates: {},kernel_inception_distance: {},time elapse for current epoch / sec : {}, ,ETA / min : {}'
        .format(epoch, loss_encoder.result(), loss_decoder.result(), loss_decoder_init.result(),loss_llh_1.result() ,kid,end_time - start_time, (epochs-epoch) *(end_time - start_time) // (60) ))

  pd.DataFrame(['logpx_1', ': ', marginal_log_likelihood_estimates_1[-1],
    # 'logpx_2', ': ', marginal_log_likelihood_estimates_2[-1],
    # 'logpx_3', ': ', marginal_log_likelihood_estimates_3[-1],
    'ess_1', ': ', ess_values_1[-1],
    # 'ess_2', ': ', ess_values_2[-1],
    # 'ess_3', ': ', ess_values_3[-1],
    ]).to_csv(os.path.join(path,'logp(x) model'),index=False)

  # Generete synthetic data
  if FLAGS.save_generated_data == True:
    synthetic_data_200 = model_0.synthesize_data(samples = 200)
    synthetic_data_500 = model_0.synthesize_data(samples = 500)
    synthetic_data_1000 = model_0.synthesize_data(samples = 1000)
    synthetic_data_2000 = model_0.synthesize_data(samples = 2000)

    np.savez_compressed(os.path.join(path,'synthetic_data_200'), synthetic_data_200)
    np.savez_compressed(os.path.join(path,'synthetic_data_500'), synthetic_data_500)
    np.savez_compressed(os.path.join(path,'synthetic_data_1000'), synthetic_data_1000)
    np.savez_compressed(os.path.join(path,'synthetic_data_2000'), synthetic_data_2000)

  #sampler_params = reshape_(sampler_params)
  if FLAGS.eval_kid == True:
    samples = train_images.shape[0]
    synthetic_data= model_0.synthesize_data(samples = samples)
    synthetic_data = tf.random.shuffle(synthetic_data)
    synthetic_data = np.array(synthetic_data)

    #fid = FID_score(test_images, synthetic_data, samples = 1000)
    kid = KID_score(train_images[:samples,::], synthetic_data,samples = samples)
    #print('FID score', fid)
    #pd.DataFrame(['FID score', ': ', fid]).to_csv(os.path.join(path,'FID score model'),index=False)
    pd.DataFrame(['KID score', ': ', kid]).to_csv(os.path.join(path,'KID score model'),index=False)

  if FLAGS.likelihood == 'Bernoulli':
    plt.clf()
    fig = plt.figure(figsize = (10, 10))
    plt.plot(encoder_losses[1:], label = "encoder_loss")
    plt.plot(decoder_losses[1:], label = "decoder_loss_MCMC")
    plt.plot(decoder_losses_init[1:], label = "decoder_loss")
    #plt.plot(KL_losses, label = 'KL-Divergence')
    #plt.plot(speed_measure_losses, label = "speed_measure_loss")
    plt.legend()
    plt.savefig(os.path.join(path,'Loss'))
  if FLAGS.likelihood == 'logistic':
    plt.clf()
    fig = plt.figure(figsize = (10, 10))
    plt.plot(encoder_losses[1:], label = "encoder_loss")
    plt.plot(decoder_losses[1:], label = "decoder_loss_MCMC")
    plt.plot(decoder_losses_init[1:], label = "decoder_loss")
    #plt.plot(KL_losses, label = 'KL-Divergence')
    #plt.plot(speed_measure_losses, label = "speed_measure_loss")
    plt.legend()
    plt.savefig(os.path.join(path,'Loss'))

  plt.clf()
  fig = plt.figure(figsize = (10, 10))
  plt.plot(kids, label = "Kernel Inception Distance")
  #plt.axhline(kid, color='r')
  plt.legend()
  plt.savefig(os.path.join(path,'Kernel Inception Distance'))


  plt.clf()
  fig = plt.figure(figsize = (10, 10))
  plt.plot(KL_losses, label = 'KL-Divergence')
  #plt.plot(speed_measure_losses, label = "speed_measure_loss")
  plt.legend()
  plt.savefig(os.path.join(path,'KL-divergance'))



if __name__ == '__main__':
 app.run(main)