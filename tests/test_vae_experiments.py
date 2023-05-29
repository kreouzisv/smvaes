##################################################
###Experiments with a Bernoulli VAE model for MNIST or FASHION-MNIST
##################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
#tf.config.list_physical_devices('GPU')
import tensorflow_probability as tfp
import numpy as np
import time
import matplotlib.pyplot as plt
from absl import app
from absl import flags
import os
import sys
from VAE_model import VAE
import pandas as pd
from load_oasis import load_oasis
#from utils import discretized_mix_logistic_loss
#from utils import sample_from_discretized_mix_logistic
from flow_based_generative_model import Flow_Generator
from utils.utils import make_bijector_kwargs
from utils.utils import sample_from_discretized_mix_logistic
from utils.utils import resnet50_encoder
#from utils import LogistixMixture
import seaborn as sns
from utils.utils import KID_score
from utils.utils import preprocess_binary_images, preprocess_images
from utils.utils import preprocess_images_logistic_mixture
from load_reduced_dataset import load_reduced_dataset
from sklearn.model_selection import train_test_split
from load_celebA import load_celebA

tfd = tfp.distributions
tfb = tfp.bijectors



#possible command line arguments
flags.DEFINE_string("sampler",
                     default="gradHMC",
                     help="MCMC sampler used: gradHMC or gradMALA or dsNUTS or dsMALA or dsHMC or HMC")

flags.DEFINE_float("learning_rate",
                     default=0.0001,
                     help="learning rate for all vae optimizers")

flags.DEFINE_float("learning_rate_mcmc",
                     default=0.001,
                     help="learning rate for all mcmc optimizers")

flags.DEFINE_integer("latent_dim",
                     default=16,
                     help="dimension of latent variables")

flags.DEFINE_integer("num_MCMC_steps",
                     default=2,
                     help="number of MCMC steps")

flags.DEFINE_integer("epochs",
                     default=100,
                     help="training epochs")

flags.DEFINE_integer("id",
                     default=0,
                     help="id of run")

flags.DEFINE_integer("num_leapfrog_steps",
                     default=2,
                     help="if HMC is used specify gradient computations")

flags.DEFINE_bool("biased_grads",
                     default=True,
                     help="true if biased log accept grads are used")

flags.DEFINE_string("netw",
                     default="mlp",
                     help="network architecture")

flags.DEFINE_string("likelihood",
                     default="Bernoulli",
                     help="Bernoulli or Normal or logistic_mix or logistic")

flags.DEFINE_string("prior",
                     default="Isotropic_Gaussian",
                     help="Isotropic_Gaussian or Vamp_prior or IAF_prior or Real_NVP_prior")

flags.DEFINE_float("beta",
                     default=1.0,
                     help="KL anealing")

flags.DEFINE_float("obs_log_var",
                     default=-2.,
                     help="if trainable")

flags.DEFINE_string("name",
                     default="",
                     help="name for identification")

flags.DEFINE_integer("reduced_sample",
                     default=0,
                     help="dataset reduction if applicable")

flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'VAE_kid_test'),
    help="Directory to put the model's fit and outputs.")
flags.DEFINE_string(
    'data_set',
    default='mnist',
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
  path=os.path.join(FLAGS.model_dir,
                    '{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}'.format(
                      flags.FLAGS.data_set,
                      flags.FLAGS.sampler,
                      flags.FLAGS.num_MCMC_steps,
                      flags.FLAGS.biased_grads, 
                      flags.FLAGS.learning_rate, 
                      flags.FLAGS.id,
                      flags.FLAGS.likelihood,
                      flags.FLAGS.prior,
                      flags.FLAGS.sampler,
                      flags.FLAGS.netw,
                      flags.FLAGS.num_leapfrog_steps,
                      flags.FLAGS.diagonal_pre_cond,
                      flags.FLAGS.latent_dim,
                      flags.FLAGS.name,
                      flags.FLAGS.reduced_sample,
                      flags.FLAGS.obs_log_var,
                      flags.FLAGS.beta))
  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()

  #set seeds to id
  tf.random.set_seed(FLAGS.id)
  np.random.seed(FLAGS.id)

  # Data Processing
  if FLAGS.data_set == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    #train_images = scale_images(train_images, (32,32,1))
    #test_images = scale_images(test_images, (32,32,1))

  if FLAGS.data_set == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

  if FLAGS.data_set == 'cifar10':
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

  if FLAGS.data_set == 'oasis':
    data, labels = load_oasis()

  if FLAGS.data_set == 'CelebA':
    train_images, test_images = load_celebA()

  if FLAGS.data_set == 'reduced_mnist':
    (__, ___), (full_test_data, __) = tf.keras.datasets.mnist.load_data()
    train_images, __, train_labels, __ = load_reduced_dataset(dataset='mnist',reduced_sample = FLAGS.reduced_sample)
    #train_images, test_images, train_label, test_labels = train_test_split(train_images, train_labels, test_size=0.00, random_state=FLAGS.id)
    #train_images = scale_images(train_images, (32,32,1))
    #test_images = scale_images(full_test_data, (32,32,1))

  if FLAGS.data_set == 'reduced_cifar':
    train_images, __, train_labels, __ = load_reduced_dataset(dataset='cifar10',reduced_sample = FLAGS.reduced_sample)
    train_images, test_images, train_label, test_labels = train_test_split(train_images, train_labels, test_size=0.20, random_state=FLAGS.id)

  if FLAGS.likelihood == 'Bernoulli':
    train_images = preprocess_binary_images(train_images)
    test_images = preprocess_binary_images(test_images)
  if FLAGS.likelihood == 'Normal':
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
  if FLAGS.likelihood == 'logistic_mix':
    train_images = preprocess_images_logistic_mixture(train_images)
    test_images = preprocess_images_logistic_mixture(test_images)
    mixtures = 10
    



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
        tf.keras.layers.Dense(400, activation = "relu"),
        tf.keras.layers.Dense(400, activation = "relu"),
        # No activation
        tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )


    decoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
        tf.keras.layers.Dense(400, activation = "relu"),
        tf.keras.layers.Dense(400, activation = "relu"),
        tf.keras.layers.Dense(np.prod(data_dim)),
        tf.keras.layers.Reshape((data_dim[0],data_dim[1],data_dim[2]))
      ]
    )
  if FLAGS.netw == 'cnn_':

    # encoder = tf.keras.Sequential([
    # tf.keras.layers.InputLayer(input_shape = data_dim),
    # tf.keras.layers.Conv2D(filters=64 , kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
    # tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
    # tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(latent_dim + latent_dim)
    # ])

    # decoder = tf.keras.Sequential([
    # tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
    # tf.keras.layers.Dense(2048),
    # tf.keras.layers.Reshape(target_shape=(4, 4, 128), input_shape=(None, 1024)),
    # tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
    # tf.keras.layers.Conv2DTranspose(filters=64 , kernel_size=4, strides=2, activation=tf.nn.relu, padding='same'),
    # tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation='linear', padding='same')
    # ])

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*128, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 128)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )



  if FLAGS.netw == 'cnn':

    encoder = tf.keras.Sequential(
      [ 
        
        tf.keras.layers.InputLayer(input_shape = data_dim),
        tf.keras.layers.Conv2D(filters = 128, kernel_size = (4, 4),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 256, kernel_size = (4, 4),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 512, kernel_size = (4, 4),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(filters = 1024, kernel_size = (4, 4),strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )


    decoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
        tf.keras.layers.Dense(units = data_dim[0]//8 * data_dim[1]//8 * 1024, activation = 'linear'),
        tf.keras.layers.Reshape(target_shape = (data_dim[0]//8, data_dim[1]//8, 1024)),
        tf.keras.layers.Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = 2,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 100, kernel_size =  (4, 4), strides = 1,padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('linear')


      ])









  if FLAGS.netw == 'resnet':

    encoder = resnet50_encoder(input_shape =data_dim,latent_dim = FLAGS.latent_dim)
    decoder = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape = (latent_dim,)),
        tf.keras.layers.Dense(units = data_dim[0]//8 * data_dim[1]//8 * 256, activation = 'linear'),
        tf.keras.layers.Reshape(target_shape = (data_dim[0]//8, data_dim[1]//8, 256)),
        tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (3, 3), strides = 2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size =  (3, 3), strides = 1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('linear')


      ])


  #######
  #Diagonal or full cholesky matrix for preconditioning
  #######

  if FLAGS.diagonal_pre_cond:

    class ConstantLayer(tf.keras.layers.Layer):
      def __init__(self):
        super(ConstantLayer, self).__init__()
        #self.constant = tf.Variable(tfd.Normal(loc=0., scale = 0.1).sample(int(latent_dim* (latent_dim+1)/2)))
        self.constant=tf.Variable(0.001 * tf.ones([latent_dim]))
                                              
      def call(self, inputs):
        return self.constant


    diag_pre_cond_nn = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = data_dim),
      tf.keras.layers.Flatten(),
      ConstantLayer()])

    def pre_cond_fn(x):
      return tf.linalg.LinearOperatorDiag(diag = diag_pre_cond_nn(x))

    pre_cond_params = diag_pre_cond_nn.trainable_variables

  else:

  

  
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


  if FLAGS.prior == 'Vamp_prior':
    # Generate Pseudoinputs
    # pseudo_inputs = PInputsGenerated(original_dim = data_dim, n_pseudo_inputs = latent_dim)
    # input_number = 100

    # pseudo_inputs = tf.Variable(
    #         initial_value=tf.random.normal((input_number, data.shape[1], data.shape[2], data.shape[3]),
    #                                        0., 1),
    #         trainable=True,
    #         #constraint=tf.keras.constraints.MinMaxNorm(0., 1.)
    #     )


    class Pseudoinputs(tf.keras.layers.Layer):
      def __init__(self):
        super(Pseudoinputs, self).__init__()

        self.pseudo_inputs = pseudo_inputs = tf.Variable(
            initial_value=tf.random.normal((1000, train_images.shape[1], train_images.shape[2], train_images.shape[3]),
                                           0., 0.001),
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(0., 1.)
        )

      def call(self):
        return self.pseudo_inputs



    pseudo_inputs = Pseudoinputs()

  else:
    pseudo_inputs = None



  if FLAGS.netw == 'IAF':

    model = Flow_Generator(latent_dim , bijector,decoder,
      learning_rate = FLAGS.learning_rate, likelihood = FLAGS.likelihood, prior = FLAGS.prior, encoder = encoder)





  else:


    model = VAE(data_dim,latent_dim, encoder, decoder, pre_cond_fn, pre_cond_params,
                             sampler = FLAGS.sampler, learning_rate = FLAGS.learning_rate,
                             num_MCMC_steps = FLAGS.num_MCMC_steps,
                             likelihood = FLAGS.likelihood, biased_grads = FLAGS.biased_grads, 
                            prior = FLAGS.prior, pseudo_inputs = pseudo_inputs, 
                            num_leapfrog_steps = FLAGS.num_leapfrog_steps, beta = FLAGS.beta, obs_log_var =FLAGS.obs_log_var)


  model.encoder.summary()
  model.decoder.summary()

  #########
  #Train the model
  #########
  epochs = FLAGS.epochs
  num_examples_to_generate = 16

  # Pick a sample of the test set for generating output images
  assert batch_size >= num_examples_to_generate
  for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]



  test_result_interval = 25

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


      # sample 
      predictions =  sample_from_discretized_mix_logistic(l= params, nr_mix = 10)
      # mean 
      #predictions = predict_from_discretized_mix_logistic(l= params, nr_mix = num_mixtures)
      




    elif FLAGS.likelihood == 'Normal':
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

    prior_gen_preds = model.synthesize_data(samples = 200)

    fig = plt.figure(figsize = (8, 10))
    for i in range(80):
      plt.subplot(10, 8, i + 1)
      #plt.imshow(prior_gen_preds[i, :, :, :])
      plt.imshow(prior_gen_preds[i, :, :, :], cmap = 'gray')

      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(os.path.join(path,'prior_generated_image_at_epoch_{:04d}.png'.format(epoch)))

  encoder_losses = []
  decoder_losses = []
  decoder_losses_init = []
  KL_losses = []

  train_marginal_likelihood_estimates = []
  test_marginal_likelihood_estimates = []
  test_marginal_likelihood_estimate_sd = []

  sampler_params = []
  grads_sampler = []
  speed_measure_losses = []
  sampler_acceptance_rates = []
  entropy_weights = []
  kids = []

  W_1 = list()
  b_1 = list()
  W_2 = list()
  b_2 = list()
  W_3 = list()
  b_3 = list()

  W_1_grads = list()
  b_1_grads = list()
  W_2_grads = list()
  b_2_grads = list()
  W_3_grads = list()
  b_3_grads = list()

  def frange_cycle_linear(n_iter, start=0.1, stop=1.0,  n_cycle=1, ratio=0.1):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L[::-1]

  betas = frange_cycle_linear(n_iter = epochs)


  for epoch in range(0, epochs + 1):
    start_time = time.time()

    # if epoch == 0.75*epochs:
    #   model.learning_rate = model.learning_rate / 10

    # if epoch % 1 == 0:
    #   try:
    #     model.beta = betas[-1]
    #     betas = betas[:-1]; betas
    #   except IndexError:
    #     continue
      #print(model.beta)
      



    #train_marginal_likelihood_estimate = tf.keras.metrics.Mean()

    for train_x in train_dataset:
      model.train_step(train_x)
      # if epoch % 1 == 0 and epoch != 0:
      #   model.beta = min(model.beta + 0.001, 1.)

  
      #model.beta = KL_annealing(epoch, model.beta)

      # if epoch % 100 == 0:
      #   train_marginal_likelihood_estimate_ = model.marginal_log_likelihood_estimate(
      #         x_test = train_x, num_particles = 128, scaling = 1.2)
      # else:
      #   train_marginal_likelihood_estimate_ = 0.

      # train_marginal_likelihood_estimate(train_marginal_likelihood_estimate_)






    

    loss_encoder = tf.keras.metrics.Mean()
    loss_decoder = tf.keras.metrics.Mean()
    loss_decoder_init = tf.keras.metrics.Mean()
    loss_llh = tf.keras.metrics.Mean()
    bits_per_dim_ = tf.keras.metrics.Mean()
    loss_speed_measure = tf.keras.metrics.Mean()
    acceptance_rate = tf.keras.metrics.Mean()
    entropy_weight = tf.keras.metrics.Mean()
    test_marginal_likelihood_estimate = tf.keras.metrics.Mean()
    loss_KL = tf.keras.metrics.Mean()




    if not epoch % test_result_interval == 0:
      continue

    for test_x in test_dataset:
      # losses
      encoder_loss_,decoder_loss_,decoder_loss_init_,KL_loss = model.compute_loss(test_x)
      loss_encoder(encoder_loss_)
      loss_decoder(decoder_loss_)
      loss_decoder_init(decoder_loss_init_)
      loss_KL(KL_loss)
      


      

      #estimate marginal likelihood

      #if epoch == epochs:
        #test_marginal_likelihood_estimate_ = model.marginal_log_likelihood_estimate(x_test = test_x, num_particles = 512, scaling = 1.0)
        #test_marginal_likelihood_estimate_ = 0.
        #test_marginal_likelihood_estimate_sd.append(test_marginal_likelihood_estimate_) 
        #test_marginal_likelihood_estimates.append(test_marginal_likelihood_estimate_)

      #test_marginal_likelihood_estimate(test_marginal_likelihood_estimate_)

      # if epoch >= sampling_init_epoch:
      # # #   #model.learning_rate = 0.0001



      #   loss_speed_measure(model.speed_measure_loss)
      #   acceptance_rate(model.acceptance_rate)
      #   entropy_weight(model.beta)


      #   W_1.append(tf.reshape(tf.convert_to_tensor(model.pre_cond_params[0]), [-1]))
      # # #   #b_1.append(tf.reshape(tf.convert_to_tensor(model.pre_cond_params[1]), [-1]))
      # # #   # W_2.append(tf.reshape(tf.convert_to_tensor(model.pre_cond_params[2]), [-1]))
      # # #   # b_2.append(tf.reshape(tf.convert_to_tensor(model.pre_cond_params[3]), [-1]))
      # # #   # W_3.append(tf.reshape(tf.convert_to_tensor(model.pre_cond_params[4]), [-1]))
      # # #   # b_3.append(tf.reshape(tf.convert_to_tensor(model.pre_cond_params[5]), [-1]))

      #   W_1_grads.append(tf.reshape(tf.convert_to_tensor(model.grads_sampler[0]), [-1]))
      #   #b_1_grads.append(tf.reshape(tf.convert_to_tensor(model.grads_sampler[1]), [-1]))
      #   # W_2_grads.append(tf.reshape(tf.convert_to_tensor(model.grads_sampler[2]), [-1]))
      #   # b_2_grads.append(tf.reshape(tf.convert_to_tensor(model.grads_sampler[3]), [-1]))
      #   # W_3_grads.append(tf.reshape(tf.convert_to_tensor(model.grads_sampler[4]), [-1]))
      #   # b_3_grads.append(tf.reshape(tf.convert_to_tensor(model.grads_sampler[5]), [-1]))

      #   speed_measure_losses.append(loss_speed_measure.result())
      #   sampler_acceptance_rates.append(acceptance_rate.result())
      #   entropy_weights.append(entropy_weight.result())






        #speed_measure_losses.append(tf.convert_to_tensor(model.speed_measure_loss))
        #sampler_acceptance_rate.append(tf.convert_to_tensor(model.acceptance_rate))
        #tf.print(np.array(speed_measure_losses).shape)

        #tf.print(model.speed_measure_loss)


    end_time = time.time()







        #sampler_params.append(W_1,b_1,W_2,b_2,W_3,b_3)

        #sampler_params.append(tf.convert_to_tensor(model.pre_cond_params))
        #grads_sampler.append(tf.convert_to_tensor(model.grads_sampler))

        #tf.print(tf.convert_to_tensor(model.grads_sampler).shape)

        #speed_measure_loss.append(tf.convert_to_tensor(model.speed_measure_loss))

        #target_log_prob.append(tf.convert_to_tensor(model.target_log_prob))



      #tf.print('model.pre_cond_params', tf.convert_to_tensor(model.pre_cond_params[-1]).shape)
      #tf.print('model.grads_sampler', tf.convert_to_tensor(model.grads_sampler[-1]).shape)
      #tf.print('model.pre_cond_params', tf.convert_to_tensor(model.pre_cond_params))
      #tf.print('model.grads_sampler', tf.convert_to_tensor(model.grads_sampler))


    encoder_losses.append(loss_encoder.result())
    decoder_losses.append(loss_decoder.result())
    decoder_losses_init.append(loss_decoder_init.result())
    test_marginal_likelihood_estimates.append(test_marginal_likelihood_estimate.result())
    KL_losses.append(loss_KL.result())
    #train_marginal_likelihood_estimates.append(train_marginal_likelihood_estimate.result())
    

    #stopEarly = Callback_EarlyStopping(marginal_likelihood_estimates_mean, min_delta=0.01, patience=20)

    # if stopEarly:
    #   print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,epochs))
    #   print("Terminating training ")
    #   break
    if epoch % 5 == 0:
      generate_and_save_images(model, epoch, test_sample)
      generate_prior_images(model, epoch)
    if epoch % 5 ==0:
      synthetic_data= np.array(model.synthesize_data(samples = 10000))
      kid = KID_score(test_images, synthetic_data,samples = 5000)
      kids.append(kid)
      

    

    print(
      'Epoch: {}, encoder_loss: {},decoder_loss: {}, decoder_loss_init: {}, test_marginal_likelihood_estimates: {},kernel_inception_distance: {},time elapse for current epoch / sec : {}, ,ETA / min : {}'
        .format(epoch, loss_encoder.result(), loss_decoder.result(), loss_decoder_init.result(),test_marginal_likelihood_estimate.result() ,kid,end_time - start_time, (epochs-epoch) *(end_time - start_time) // (60) ))


      





  # evaluate model
  # batch_size_eval = 16
  # test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
  #                 .shuffle(test_size).batch(batch_size_eval))

  # for test_x in tqdm(test_dataset):
  #   test_marginal_likelihood_estimate_ = model.marginal_log_likelihood_estimate(x_test = test_x, num_particles = 512, scaling = 1.0)
  #   test_marginal_likelihood_estimates.append(test_marginal_likelihood_estimate_)






  # maybe random sample 1000 test images
  # batch_size = 3500
  # test_data = (tf.data.Dataset.from_tensor_slices(test_images)
  #                 .shuffle(test_size).batch(batch_size))

  # batched_estimates = []
  # for test_data in test_data:
  #   ais_estimate = model.marginal_likelihood_estimate_ais(test_data)
  #   print('ais estimate', ais_estimate)
  #   batched_estimates.append(ais_estimate)

  # ais_estimate = tf.reduce_mean(batched_estimates)

  # Generete synthetic data
  # synthetic_data_200 = model.synthesize_data(samples = 200)
  # synthetic_data_500 = model.synthesize_data(samples = 500)
  # synthetic_data_1000 = model.synthesize_data(samples = 1000)
  # synthetic_data_2000 = model.synthesize_data(samples = 2000)
  #synthetic_data_10000 = model.synthesize_data(samples = 10000)


  # fig = plt.figure(figsize = (8, 10))
  # for i in range(80):
  #   plt.subplot(10, 8, i + 1)
  #   plt.imshow(tf.cast(synthetic_data_500[i, :, :, :] * 255, tf.uint8), cmap = 'gray')
  #   plt.axis('off')

  #   # tight_layout minimizes the overlap between 2 sub-plots
  # plt.savefig(os.path.join(path,'generated_images_with_prior'))


  #np.savez_compressed(os.path.join(path,'synthetic_images_10000'), synthetic_data_10000)
  # np.savez_compressed(os.path.join(path,'synthetic_images_500'), synthetic_data_500)
  
  

  #print('target_log_prob_shape', target_log_prob.shape)

  def reshape_(x):
    x = np.array(x)
    x = x.reshape(-1, x.shape[-1])
    x_mean = tf.reduce_mean(x, axis = -1)
    x_std = tf.math.reduce_std(x, axis = -1)
    x_ci = 1.96 * x_std/np.sqrt(x_mean.shape[-1])

    return x, x_std

  #W_1, W_1_sd  = reshape_(W_1)
  # b_1, b1_sd = reshape_(b_1)
  # W_2, W_2_sd = reshape_(W_2)
  # b_2, b_2_sd = reshape_(b_2)
  # W_3, W_3_sd = reshape_(W_3)
  # b_3, b_3_sd = reshape_(b_3)

  

  #W_1_grads, W_1_grads_sd = reshape_(W_1_grads)
  # b_1_grads, b_1_grads_sd = reshape_(b_1_grads)
  # W_2_grads, W_2_grads_sd  = reshape_(W_2_grads)
  # b_2_grads, b_2_grads_sd = reshape_(b_2_grads)
  # W_3_grads, W_3_grads_sd = reshape_(W_3_grads)
  # b_3_grads, b_3_grads_sd = reshape_(b_3_grads)


  # weights = np.stack([W_1,b_1,W_2,b_2,W_3,b_3], axis = 0)
  # weights_ci = np.stack([W_1_sd,b1_sd,W_2_sd,b_2_sd,W_3_sd,b_3_sd], axis = 0)

  # grads = np.stack([W_1_grads,b_1_grads,W_2_grads,b_2_grads,W_3_grads,b_3_grads], axis = 0)
  # grads_ci = np.stack([W_1_grads_sd,b_1_grads_sd,W_2_grads_sd,b_2_grads_sd,W_3_grads_sd,b_3_grads_sd], axis = 0)


  # tf.print('weights_shape', weights.shape)
  # tf.print('weights_ci_shape', weights_ci.shape)
  # tf.print('grads_shape', grads.shape)
  # tf.print('grads_ci_shape', grads_ci.shape)


  #sampler_params = reshape_(sampler_params)

  synthetic_data= model.synthesize_data(samples = 10000)
  synthetic_data = tf.random.shuffle(synthetic_data)
  synthetic_data = np.array(synthetic_data)

  #fid = FID_score(test_images, synthetic_data, samples = 1000)
  kid = KID_score(test_images, synthetic_data,samples = 10000)
  #print('FID score', fid)
  #pd.DataFrame(['FID score', ': ', fid]).to_csv(os.path.join(path,'FID score model'),index=False)
  pd.DataFrame(['KID score', ': ', kid]).to_csv(os.path.join(path,'KID score model'),index=False)


  # plt.clf()
  # fig = plt.figure(figsize = (10, 10))
  # for i in range(layers):
  #   plt.subplot(2, 3, i + 1)
  #   plt.plot(np.arange(weights.shape[-1]),  weights[i,:])
  #   plt.fill_between(np.arange(weights.shape[-1]), (weights[i,:]-weights_ci[i,:]), (weights[i,:]+weights_ci[i,:]), color='b', alpha=.1)
  #   plt.xlabel('iterations')
  #   plt.ylabel(f'layer{i}')
  # fig.tight_layout()
  # plt.savefig(os.path.join(path,'Weights'))



  # plt.clf()
  # fig = plt.figure(figsize = (10, 10))
  # for i in range(layers):
  #   plt.subplot(2, 3, i + 1)
  #   plt.plot(np.arange(grads.shape[-1]),grads[i,:])
  #   plt.fill_between(np.arange(grads.shape[-1]), (grads[i,:]-grads_ci[i,:]), (grads[i,:]+grads_ci[i,:]), color='b', alpha=.1)
  #   plt.ylim(3 * -10**3, 3 * 10**3)
  #   plt.xlabel('iterations')
  #   plt.ylabel(f'layer{i}')
  # fig.tight_layout()
  # plt.savefig(os.path.join(path,'Grads'))


  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(W_1, label = "W_1")
  # plt.legend()
  # plt.savefig(os.path.join(path,'W_1'))
  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(W_1_grads, label = "W_1")
  # plt.ylim(3 * -10**3, 3 * 10**3)
  # plt.legend()
  # plt.savefig(os.path.join(path,'W_1_Grads'))

  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(b_1, label = "b_1")
  # plt.legend()
  # plt.savefig(os.path.join(path,'b_1'))
  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(b_1_grads, label = "b_1")
  # plt.ylim(3 * -10**3, 3 * 10**3)
  # plt.legend()
  # plt.savefig(os.path.join(path,'b_1_Grads'))

  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(W_2, label = "W_2")
  # plt.legend()
  # plt.savefig(os.path.join(path,'W_2'))
  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(W_2_grads, label = "W_2")
  # plt.ylim(3 * -10**3, 3 * 10**3)
  # plt.legend()
  # plt.savefig(os.path.join(path,'W_2_Grads'))


  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(b_2, label = "b_2")
  # plt.legend()
  # plt.savefig(os.path.join(path,'b_2'))
  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(b_2_grads, label = "b_2")
  # plt.ylim(3 * -10**3, 3 * 10**3)
  # plt.legend()
  # plt.savefig(os.path.join(path,'b_2_Grads'))

  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(W_3, label = "W_3")
  # plt.legend()
  # plt.savefig(os.path.join(path,'W_3'))
  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(W_3_grads, label = "W_3")
  # plt.ylim(3 * -10**3, 3 * 10**3)
  # plt.legend()
  # plt.savefig(os.path.join(path,'W_3_Grads'))


  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(b_3, label = "b_3")
  # plt.legend()
  # plt.savefig(os.path.join(path,'b_3'))
  # plt.clf()
  # fig = plt.figure(figsize = (15, 15))
  # plt.clf()
  # plt.plot(b_3_grads, label = "b_3")
  # plt.ylim(3 * -10**3, 3 * 10**3)
  # plt.legend()
  # plt.savefig(os.path.join(path,'b_3_Grads'))


  # plt.clf()
  # fig = plt.figure(figsize = (20, 20))
  # for i in range(size):
  #   plt.subplot(5, 11, i + 1)
  #   plt.plot(sampler_params[:,:,:])
  # fig.tight_layout()
  # plt.savefig(os.path.join(path,'Sampler_params'))



  # plt.clf()
  # fig = plt.figure(figsize = (20, 20))
  # for i in range(layers):
  #   plt.subplot(5, 11, i + 1)
  #   plt.plot(grads_sampler[:,:,:,i])
  #   plt.ylim(-10**4, 10**4)
  # fig.tight_layout()
  # plt.savefig(os.path.join(path,'Sampler_grads'))

  # plt.clf()
  # fig = plt.figure(figsize = (20, 20))
  # for i in range(size):
  #   plt.subplot(5, 11, i + 1)
  #   plt.plot(speed_measure_loss[:,:,i])
  # fig.tight_layout()
  # plt.savefig(os.path.join(path,'speed_measure_loss'))

  
  # for i in range(FLAGS.latent_dim):
  #   sampler_params_log = pd.DataFrame(sampler_params[:,:,i])
  #   sampler_params_log.to_csv(os.path.join(path,f'sampler_params_log{i}'), index = False)

  #   grads_sampler_log = pd.DataFrame(grads_sampler[:,:,i])
  #   grads_sampler_log.to_csv(os.path.join(path,f'grads_sampler_log{i}'), index = False)

  #speed_measure_losses = np.reshape(speed_measure_losses, [-1])

  #print('speed_measure_losses', np.array(speed_measure_losses).shape)

  #speed_measure_losses = np.reshape(speed_measure_losses, ((epochs - sampling_init_epoch + 10) // test_result_interval * (test_size // batch_size_test), x.shape[-1]))

  plt.clf()
  fig = plt.figure(figsize = (10, 10))
  plt.plot(encoder_losses, label = "encoder_loss")
  plt.plot(decoder_losses, label = "decoder_loss_MCMC")
  plt.plot(decoder_losses_init, label = "decoder_loss")
  plt.plot(KL_losses, label = 'KL-Divergence')
  #plt.plot(speed_measure_losses, label = "speed_measure_loss")
  plt.legend()
  plt.savefig(os.path.join(path,'Loss'))

  plt.clf()
  fig = plt.figure(figsize = (10, 10))
  plt.plot(kids, label = "Kernel Inception Distance")
  plt.axhline(kid, color='r')
  plt.legend()
  plt.savefig(os.path.join(path,'Kernel Inception Distance'))

  # plt.clf()
  # fig = plt.figure(figsize = (10, 10))
  # plt.plot(sampler_acceptance_rates)
  # plt.xlabel('iterations')
  # plt.ylabel('Mean Acceptance Rate')
  # plt.ylim(0, 1)
  # plt.savefig(os.path.join(path,'Acceptance_Rate'))

  # plt.clf()
  # fig = plt.figure(figsize = (10, 10))
  # plt.plot(entropy_weights)
  # plt.xlabel('iterations')
  # plt.ylabel('Mean Entropy Weight')
  # plt.savefig(os.path.join(path,'Entropy_Weight'))


  # plt.clf()
  # fig = plt.figure(figsize = (10, 10))
  # plt.plot(speed_measure_losses)
  # plt.xlabel('iterations')
  # plt.ylabel('Mean mcmc objective')
  # #plt.legend()
  # plt.savefig(os.path.join(path,'Speed_Measure_Loss'))

  plt.clf()
  #plt.hist(test_marginal_likelihood_estimates)
  #sns.kdeplot(np.array(test_marginal_likelihood_estimates))
  sns.histplot(np.array(test_marginal_likelihood_estimates))
  #plt.axvline(np.mean(test_marginal_likelihood_estimates), color='k', linestyle='dashed', linewidth=1)
  #min_ylim, max_ylim = plt.ylim()
  #plt.text(np.mean(test_marginal_likelihood_estimates)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(test_marginal_likelihood_estimates)))
  plt.savefig(os.path.join(path,'log p(x)'))


  pd.DataFrame(['logpx', ': ', np.mean(test_marginal_likelihood_estimates)]).to_csv(os.path.join(path,'logp(x) model'),index=False)




if __name__ == '__main__':
 app.run(main)