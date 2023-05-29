##################################################
###Experiments with a linear VAE model on simulated data and different covariance structures
##################################################

import tensorflow as tf
import tensorflow_probability as tfp
import time
import Gaussian_VAE

tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import time
import os
from absl import app
from absl import flags
import sys
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

dtype=tf.float32

flags.DEFINE_string("cov_type",
                     default="independent",
                     help="cov_type of Gaussian target")
flags.DEFINE_integer("num_steps",
                     default=5,
                     help="number of MCMC steps")
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'linear_VAE'),
    help="Directory to put the model's fit.")
flags.DEFINE_float("learning_rate",
                     default=.01,
                     help="learning rate for all optimizers")
flags.DEFINE_integer("id",
                     default=0,
                     help="id of run")

FLAGS = flags.FLAGS


if True:
  FLAGS(sys.argv)

#def main(argv):
#  del argv  # unused

  #save (command line) flags to file
  fv = flags._flagvalues.FlagValues()
  key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
  s = '\n'.join(f.serialize() for f in key_flags)
  print('specified flags:\n{}'.format(s))
  path=os.path.join(FLAGS.model_dir,
                    '_{}_covariance__{}_steps__{}'.format(
                      flags.FLAGS.cov_type,flags.FLAGS.num_steps,
                      flags.FLAGS.id))
  if not os.path.exists(path):
    os.makedirs(path)
  flag_file = open(os.path.join(path,'flags.txt'), "w")
  flag_file.write(s)
  flag_file.close()


  ##################
  #Generate synthetic data set
  if FLAGS.cov_type == 'independent':
    dims = 4
    num_samples = 2000
    W_true = tf.cast(tf.linalg.diag(np.concatenate([np.linspace(.1, 1., 2), np.linspace(dims+1, 3*dims, 2)])), dtype)
    true_observation_noise_std = 1.
    true_latents = tfd.Normal(loc=tf.zeros([dims]), scale = tf.ones([dims])).sample(num_samples)
    observation_noise = tfd.Normal(loc=tf.zeros([dims]), scale = tf.ones([dims])).sample(num_samples)
    data = tf.linalg.matvec(W_true, true_latents) + observation_noise

    #define a transformation that is only a diagonal matrix
    latent_dim = 4


  scale_tril = tf.Variable(initial_value = tf.eye(latent_dim))
  bijector = tfb.ScaleMatvecTriL(scale_tril = scale_tril)
  bijector_params = [bijector.parameters['scale_tril']]




  ########
  # Define linear VAE
  ########
  
  # encoder and decoder are just affine functions
  encoder = tf.keras.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape = (dims)),
      tf.keras.layers.Flatten(),
      # No activation
      tf.keras.layers.Dense(latent_dim + latent_dim),
    ]
  )
  # just output the mean param
  decoder = tf.keras.Sequential(
    [
      tf.keras.layers.InputLayer(input_shape = (latent_dim)),
      tf.keras.layers.Dense(dims, activation = "linear"),
      #tf.keras.layers.Reshape((dims))
    ]
  )
  # decoder = tf.keras.Sequential(
  #   [
  #     tf.keras.layers.InputLayer(input_shape = (latent_dim)),
  #     tf.keras.layers.Dense(latent_dim, activation = "linear"),
  #     tf.keras.layers.Reshape((1, latent_dim, 1))
  #   ]
  # )



  model = Gaussian_VAE.GaussianVAE(latent_dim, encoder, decoder, bijector, bijector_params, learning_rate = FLAGS.learning_rate, num_MCMC_steps = FLAGS.num_steps)


  #########
  #Train the model with adaptation of the transformed MALA kernel
  #########
  epochs = 100
  mini_batch_size = 50
  encoder_losses = []
  decoder_losses = []
  decoder_losses_init = []
  held_out_log_likelihoods = []
  params = []
  for epoch in range(1, epochs + 1):
    for train_x in tf.split(data, num_samples//mini_batch_size):
      #model.train_step(tf.expand_dims(train_x, 1))
      model.train_step(train_x)
      loss_encoder = tf.keras.metrics.Mean()
      loss_decoder = tf.keras.metrics.Mean()
      loss_decoder_init = tf.keras.metrics.Mean()
      p_x_mean = tf.keras.metrics.Mean()

    if epoch in range(0,100,5):
      for test_x in tf.split(data,1):
        #params.append(scale_diag.numpy())
        # losses
        encoder_loss = loss_encoder(model.compute_loss(test_x)[0])
        decoder_loss = loss_decoder(model.compute_loss(test_x)[1])
        decoder_loss_init = loss_decoder_init(model.compute_loss(test_x)[2])
        p_x = p_x_mean(model.compute_loss(test_x)[3])

        encoder_losses.append(loss_encoder.result().numpy())
        decoder_losses.append(loss_decoder.result().numpy())
        decoder_losses_init.append(loss_decoder_init.result().numpy())

  plt.plot(encoder_losses)
  plt.savefig(os.path.join(path,'encoder_losses.png'))
  plt.close('all')
  plt.plot(decoder_losses)
  plt.savefig(os.path.join(path,'decoder_losses.png'))
  plt.close('all')
  plt.plot(decoder_losses_init)
  plt.savefig(os.path.join(path,'decoder_losses_init_init.png'))
  plt.close('all')
  #plt.plot(np.stack(params))
  #plt.savefig(os.path.join(path,'transformation_params.png'))
  #plt.close('all')
  #
#if __name__ == '__main__':
#  app.run(main)

  