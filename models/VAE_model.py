from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors
import warnings
warnings.filterwarnings('ignore')
#from utils import logistic_mixture_log_likelihood
#from utils import pixelcnn_loss
#from utils import corelated_covar
#from utils import discretized_mix_logistic_loss_ as discretized_mix_logistic_loss_

#from adaptiveMALA import AdaptiveMALA
from samplers.grad_adaptive_MALA import AdaptiveMALA
from samplers.grad_adaptive_hmc import AdaptiveHMC

from utils.utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic


# Define log_likelihood for Normal dist.

def log_logistic256(x, mean, logvar, reduce_dim=None, name=None):
  """
  Discretized log-logistic. Similar to log-normal, but with heavier tails.
  @param reduce_dim: dimension of the data attributes, along which to sum the log-prob.
      If tensor has shape (N, sample_size) then provide reduce_dim=1
      If tensor has shape (N, L, sample_size) then provide reduce_dim=2
  """
  binsize = 1. / 256.
  scale = tf.math.exp(logvar)
  #tf.print('x / binsize * binsize',((x / binsize) * binsize))
  x_std = (tf.math.floor(x / binsize) * binsize - mean) / scale
  #tf.print('x_std',x_std[0])
  logp = tf.math.log(tf.sigmoid(x_std + binsize / scale) - tf.sigmoid(x_std) + 1e-7)

  return tf.reduce_sum(logp, axis=reduce_dim, name=name)

def log_normal_pdf(sample, mean, logvar, raxis = 1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis = raxis)

def log_normal_diag(x, mean, logvar, reduce_dim=None, name=None):
    """
    Multivariate log normal
    @param reduce_dim: dimension of the data attributes, along which to sum the log-prob.
        If tensor has shape (minibatch, sample_size) then provide reduce_dim=1
        If tensor has shape (N, L, sample_size) then provide reduce_dim=2
    """
    log2pi = np.log(2 * np.pi)
    log_normal = -.5 * (log2pi + logvar + tf.math.pow(x - mean, 2) / tf.math.exp(logvar))
    return tf.reduce_sum(log_normal, axis=reduce_dim, name=name)


class VAE(tf.keras.Model):
  """general variational autoencoder class."""

  def __init__(self, data_dim,latent_dim, encoder, decoder,  pre_cond_fn, pre_cond_params, likelihood, sampler = "gradMALA",
               learning_rate=1e-4, learning_rate_mcmc=1e-3,num_MCMC_steps=5,  num_leapfrog_steps = 2, beta = 1.,
               min_entropy_weight = .001, max_entropy_weight = 100, biased_grads = True,
               prior = None, observation_std = 1., prior_correlation = 0.0,
               mixture_components = 10, pseudo_inputs = None, obs_log_var = 0.
               ):

    super(VAE, self).__init__()
    self.latent_dim = latent_dim
    self.data_dim = data_dim
    self.encoder = encoder
    self.decoder = decoder
    self.pre_cond_fn = pre_cond_fn
    self.pre_cond_params = pre_cond_params
    self.num_leapfrog_steps = num_leapfrog_steps
    self.optimizer_decoder = tf.keras.optimizers.Adam(learning_rate)
    self.optimizer_encoder = tf.keras.optimizers.Adam(learning_rate)
    self.optimizer_entropy_weight = tf.keras.optimizers.Adam(10 * learning_rate)
    self.optimizer_mcmc = tf.keras.optimizers.Adam(learning_rate_mcmc)
    self.num_MCMC_steps = num_MCMC_steps
    self.proposal_entropy_weight = tf.Variable(initial_value = 1., trainable = True)
    self.penalty_param = tf.Variable(initial_value = 1000., trainable = True)
    self.min_entropy_weight = min_entropy_weight
    self.max_entropy_weight = max_entropy_weight
    self.learning_rate = learning_rate
    self.learning_rate_mcmc = learning_rate_mcmc
    self.sampler = sampler
    self.likelihood = likelihood
    self.biased_grads = biased_grads
    self.pseudo_inputs = pseudo_inputs
    self.optmizer_pseudo_inputs = tf.keras.optimizers.Adam(learning_rate * 100)
    self.optimizer_prior = tf.keras.optimizers.Adam(learning_rate // 10)
    self.prior_mixtures = 50
    self.prior_mean = tf.Variable(tf.random.normal(shape=[self.prior_mixtures , self.latent_dim]), trainable = True)
    self.prior_log_var = tf.Variable(tf.ones(shape=[self.prior_mixtures, self.latent_dim]), trainable = True)
    #self.observation_std = observation_std
    self.step_size_log = 1.0
    self.observation_log_var = tf.Variable(initial_value = obs_log_var, trainable = True, constraint=lambda z: tf.clip_by_value(z, -4.5, 4.5))
    self.optimizer_observation_noise = tf.keras.optimizers.Adam(10*learning_rate)
    self.prior_correlation = prior_correlation
    self.beta = tf.Variable(initial_value = beta, trainable = True)
    self.mixture_components = mixture_components

    self.prior = prior

    if self.prior == 'Isotropic_Gaussian':
      prior_dist = tfd.MultivariateNormalDiag(scale_diag=tf.ones([latent_dim]))

    if self.prior == 'Vamp_prior':
      pseudo_mean, pseudo_logvar = self.encode(self.pseudo_inputs(None))

      prior_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(
                probs=1.0/self.pseudo_inputs.get_n() * tf.ones(self.pseudo_inputs.get_n())
            ),
            components_distribution=tfd.MultivariateNormalDiag(pseudo_mean,tf.math.exp(pseudo_logvar)),
        )

    if self.prior == 'Gaussian_Mixture':

      def get_prior(num_modes, prior_mean, prior_log_var):
        prior = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[1 / num_modes,] * num_modes),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=prior_mean,
                scale_diag= tf.math.exp(prior_log_var)
            )
        )

        return prior

      prior_dist = get_prior(self.prior_mixtures, self.prior_mean, self.prior_log_var)

    if self.prior == 'IAF_prior':

      hidden_layers = [1024, 1024]


      self.prior_params = tfb.AutoregressiveNetwork(
        params=2,
        hidden_units=hidden_layers,
        event_shape=self.latent_dim,
        # conditional=True,
        kernel_initializer=tf.keras.initializers.VarianceScaling(0.01),
        # conditional_event_shape=np.prod(self.data_dim)
      )

      prior_dist = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim)),
      bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(self.prior_params)))

    if self.prior == 'Real_NVP_prior':

      hidden_layers = [1024, 1024]

      self.prior_params = tfb.AutoregressiveNetwork(
        params=2,
        hidden_units=hidden_layers,
        event_shape=self.latent_dim,
        kernel_initializer=tf.keras.initializers.VarianceScaling(0.01),
      )

      prior_dist = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim)),
      bijector=tfb.RealNVP(
        num_masked=2,
        shift_and_log_scale_fn=tfb.real_nvp_default_template(
            hidden_layers=[512, 512])))

      # prior_dist = tfd.TransformedDistribution(
      # distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim)),
      # bijector=tfb.Invert(tfb.RealNVP(self.prior_params)))


    self.prior_dist = prior_dist



  def recompute_prior(self,prior_mean, prior_log_var,z=None, samples=0):
    # if prior is trainable, has to be recomputed for gradient evaluations

    if self.prior == 'Gaussian_Mixture':


      prior = tfd.MixtureSameFamily(
              mixture_distribution=tfd.Categorical(probs=[1 / self.prior_mixtures ,] * self.prior_mixtures ),
              components_distribution=tfd.MultivariateNormalDiag(
                  loc=prior_mean,
                  scale_diag= tf.math.exp(prior_log_var)
              )
          )

      if samples > 0:
        return prior.sample(samples)

      prior_log_prob = prior.log_prob(z)



    if self.prior == 'Vamp_prior':

      prior = tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(
              probs=1.0/self.pseudo_inputs.get_n() * tf.ones(self.pseudo_inputs.get_n())
          ),
          # prior_mean and prior_log_var are pseudo_mean and pseudo_log_var
          components_distribution=tfd.MultivariateNormalDiag(prior_mean,tf.math.exp(prior_log_var)),
      )

      if samples > 0:
        return prior.sample(samples)

      prior_log_prob = prior.log_prob(z)

    return prior_log_prob


  #log target function
  def target_log_prob_fn(self,x):
      #return lambda z: self.prior_dist(z = z,pseudo_inputs = self.pseudo_inputs.call(), C = 100) + self.llh(z,x)
      return lambda z: self.prior_dist.log_prob(z) + self.llh(z,x)
      #return lambda z: self.prior_dist(z = z,pseudo_inputs = self.pseudo_inputs.call(), C = 100) + self.llh(z,x)


  def llh(self, z, x, eval = False):

    if self.likelihood == 'Bernoulli':
      x_logit = self.decode(z)
      decoder_llh = tf.reduce_sum(tfd.Bernoulli(logits = x_logit).log_prob(x), axis = [1,2,3])
      return decoder_llh

    elif self.likelihood == 'Normal':

      x_mean, x_log_var = self.decode(z)
      #tf.print(x_log_var)
      #tf.print(self.observation_log_var)
      #decoder_llh = tf.reduce_sum(tfd.Normal(loc = x_mean, scale = tf.math.exp(self.observation_log_var)).log_prob(x), axis=[1,2,3])
      decoder_llh = log_normal_pdf(x_mean, x, x_log_var, raxis = [1,2,3])
      #decoder_llh = tf.reduce_sum(tfd.Normal(loc = x_mean, scale = self.observation_std).log_prob(x), axis=[1,2,3])

      return decoder_llh

    elif self.likelihood == 'Log_Normal':

      x_mean, x_log_var = self.decode(z)
      decoder_llh = tf.reduce_sum(tfd.LogLogistic(loc = x_mean, scale = tf.math.exp(self.observation_log_var)).log_prob(x), axis=[1,2,3])

      return decoder_llh

    elif self.likelihood == 'logistic_mix':


      params = self.decode(z)
      ll = tf.reduce_sum(discretized_mix_logistic_loss(x=x,l=params), axis = [-1,-2])
      #print(ll.shape)

      #print(discretized_mix_logistic_loss(x=x,l=params))

      #ll = LogistixMixture.log_prob(x, params)

      
      return ll

    elif self.likelihood == 'logistic':
      x_mean, x_log_var = self.decode(z)
      ll = log_logistic256(x, x_mean, x_log_var, [1,2,3])

      return ll

    elif self.likelihood == 'Categorical':

      logits = self.decode(z)

      #import pdb
      #pdb.set_trace()
      x = tf.squeeze(x, axis = -1)
      # Note that tfp.categorical perform expand dims axis -1 on x when log_prob is called
      ll = tf.reduce_sum(tfp.distributions.Categorical(logits).log_prob(x), axis=[1,2])

      return ll

  def encode(self, x):
    #return mean and log variance of initial variaitional distribution
    
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits = 2, axis = 1)
    #logvar = tf.math.softplus(logvar)
    

    return mean, logvar

  def reparameterize_initial_sample(self, mean, logvar, eps):
    #function to reparamterise samples from the initial variational distribution
    initial_state = eps * tf.exp(logvar * .5) + mean
    return initial_state


  def apply_MCMC_kernels(self, init_state, target_log_prob_fn, x, training = True):
    #apply MCMC kernels kernels
    #consider MALA kernels on a transformed space
    #this allows for some preconditioning beyond a diagonal transformation
    #we use the transformed state as the sample from the initial variational distribution
    # Sampler Dictironary :
    # gradMALA : Gradient Based Adaptive MALA
    # dsNUTS : Dual Averaging Step Size Adaptation No U Turn Sampler
    # dsMALA : Dueal Averaging Step Size Adaptation MALA
    sampler = self.sampler

    if sampler == "gradHMC":

      

      kernel = AdaptiveHMC(
        num_leapfrog_steps = self.num_leapfrog_steps,
        target_log_prob_fn = target_log_prob_fn,
        pre_cond_fn = self.pre_cond_fn,
        pre_cond_params = self.pre_cond_params,
        x = x,
        learning_rate_beta = self.learning_rate_mcmc,
        proposal_entropy_weight = self.proposal_entropy_weight,
        penalty_param = self.penalty_param,
        min_entropy_weight = self.min_entropy_weight,
        max_entropy_weight = self.max_entropy_weight,
        biased_grads = self.biased_grads,
        optimizer = self.optimizer_entropy_weight,
      )

      states, kernel_results = tfp.mcmc.sample_chain(
        num_results = self.num_MCMC_steps,
        current_state = init_state,
        previous_kernel_results = kernel.bootstrap_results(init_state = init_state),
        kernel = kernel,
        num_burnin_steps = 0)










      return states, kernel_results

    if sampler == "gradMALA":

      kernel = AdaptiveMALA(
        target_log_prob_fn = target_log_prob_fn,
        pre_cond_fn = self.pre_cond_fn,
        pre_cond_params = self.pre_cond_params,
        x = x,
        learning_rate_beta = self.learning_rate,
        proposal_entropy_weight = self.proposal_entropy_weight,
        min_entropy_weight = self.min_entropy_weight,
        max_entropy_weight = self.max_entropy_weight,
        biased_grads = self.biased_grads,
        optimizer = self.optimizer_mcmc
      )


      states, kernel_results = tfp.mcmc.sample_chain(
        num_results = self.num_MCMC_steps,
        current_state = init_state,
        previous_kernel_results = kernel.bootstrap_results(init_state = init_state),
        kernel = kernel,
        num_burnin_steps = 0)




      # self.grads_sampler = kernel._impl.inner_kernel.grads
      # #self.target_log_prob = kernel._impl.inner_kernel.target_log_prob
      # self.speed_measure_loss = kernel._impl.inner_kernel.speed_measure_loss
      # self.acceptance_rate = kernel._impl.inner_kernel.acceptance_rate
      # self.beta = kernel._impl.inner_kernel.beta




      return states, kernel_results
    
    if sampler == "dsNUTS":
    
      sampler = tfp.mcmc.TransformedTransitionKernel(
      tfp.mcmc.NoUTurnSampler(
          target_log_prob_fn=target_log_prob_fn,
          step_size=0.1),
          bijector=tfp.bijectors.Identity())
    
      adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
          inner_kernel=sampler,
          num_adaptation_steps=self.num_MCMC_steps,
          target_accept_prob=0.65,
          step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
          inner_results=pkr.inner_results._replace(step_size=new_step_size)
          ),
          step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
          log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
          )
    
      def do_sampling():
        return tfp.mcmc.sample_chain(
        kernel=adaptive_sampler,
        current_state=init_state,
        num_results=self.num_MCMC_steps,
        num_burnin_steps=0)
    
      states, kernel_results = do_sampling()
    
      return states, kernel_results

    if sampler == "MALA":
    # Use the hmc with single leap frog to recover MALA https://arxiv.org/pdf/1905.12247.pdf
      sampler = tfp.mcmc.TransformedTransitionKernel(
          tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          num_leapfrog_steps=1,
          step_size=1),
          bijector=tfp.bijectors.Identity())
    
      def do_sampling():
        return tfp.mcmc.sample_chain(
        kernel=sampler,
        current_state=init_state,
        num_results=self.num_MCMC_steps,
        num_burnin_steps=0)
    
      states, kernel_results = do_sampling()

      #self.grads_sampler = kernel_results.inner_results.inner_results.proposed_results.grads_target_log_prob
      #self.target_log_prob = kernel_results.inner_results.inner_results.proposed_results.target_log_prob
    
      return states, kernel_results
    
    if sampler == "dsMALA__":
    # Use the hmc with single leap frog to recover MALA https://arxiv.org/pdf/1905.12247.pdf
      sampler = tfp.mcmc.TransformedTransitionKernel(
          tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          num_leapfrog_steps=1,
          step_size=1.0),
          bijector=tfp.bijectors.Identity())
    
      adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
          inner_kernel=sampler,
          num_adaptation_steps=self.num_MCMC_steps,
          target_accept_prob=0.55
          )
    
      def do_sampling():
        return tfp.mcmc.sample_chain(
        kernel=adaptive_sampler,
        current_state=init_state,
        num_results=self.num_MCMC_steps,
        num_burnin_steps=0)
    
      states, kernel_results = do_sampling()

      #self.grads_sampler = kernel_results.inner_results.inner_results.proposed_results.grads_target_log_prob
      #self.target_log_prob = kernel_results.inner_results.inner_results.proposed_results.target_log_prob
    
      return states, kernel_results

    if sampler == "dsMALA":

      # step size is only adapted during training


      if training == True:
        #tf.print('adapting')


        sampler = tfp.mcmc.TransformedTransitionKernel(
          tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          num_leapfrog_steps=1,
          step_size=1.0,store_parameters_in_results = True),
          bijector=tfp.bijectors.Identity())

    
        adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=sampler,
            num_adaptation_steps=self.num_MCMC_steps,
            target_accept_prob=0.55
            )

        states, kernel_results = tfp.mcmc.sample_chain(
          kernel=adaptive_sampler,
          current_state=init_state,
          previous_kernel_results = adaptive_sampler.bootstrap_results(init_state = init_state),
          num_results=self.num_MCMC_steps,
          num_burnin_steps=0)

        # log step size of final sample to use for testing
        self.step_size_log = kernel_results.new_step_size[-1]

        #tf.print(kernel_results)



    
      else:
        #tf.print('evaluating')
        

        sampler = tfp.mcmc.TransformedTransitionKernel(
          tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          num_leapfrog_steps=1,
          step_size=self.step_size_log,store_parameters_in_results = True),
          bijector=tfp.bijectors.Identity())

        states, kernel_results = tfp.mcmc.sample_chain(
          kernel=sampler,
          current_state=init_state,
          num_results=self.num_MCMC_steps,
          num_burnin_steps=0)


        


      #tf.print(self.step_size_log)
      # tf.print(tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
      # kernel_results.inner_results.inner_results.accepted_results.log_accept_ratio, 0.))))
      # tf.print(tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
      # kernel_results.log_accept_ratio, 0.))))

      #tf.print(kernel_results)

      


    
      return states, kernel_results

    if sampler == "HMC":
    # Use the hmc with single leap frog to recover MALA https://arxiv.org/pdf/1905.12247.pdf
      sampler = tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          num_leapfrog_steps=self.num_leapfrog_steps,
          step_size=1)
          
    
      
      def do_sampling():
        return tfp.mcmc.sample_chain(
        kernel=sampler,
        current_state=init_state,
        num_results=self.num_MCMC_steps,
        num_burnin_steps=0)
    
      states, kernel_results = do_sampling()


      return states, kernel_results


    if sampler == "dsHMC":

      # step size is only adapted during training


      if training == True:
        #tf.print('adapting')


        sampler = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=self.num_leapfrog_steps,
            step_size=1.0,store_parameters_in_results = True),
            bijector=tfp.bijectors.Identity())

      
        adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
              inner_kernel=sampler,
              num_adaptation_steps=self.num_MCMC_steps,
              target_accept_prob=0.65
              )

        states, kernel_results = tfp.mcmc.sample_chain(
            kernel=adaptive_sampler,
            current_state=init_state,
            #previous_kernel_results = adaptive_sampler.bootstrap_results(init_state = init_state),
            num_results=self.num_MCMC_steps,
            num_burnin_steps=0)

          # log step size of final sample to use for testing
        self.step_size_log = kernel_results.new_step_size[-1]

        #tf.print(kernel_results)



    
      else:
        #tf.print('evaluating')
        

        sampler = tfp.mcmc.TransformedTransitionKernel(
          tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          num_leapfrog_steps=self.num_leapfrog_steps,
          step_size=self.step_size_log,store_parameters_in_results = True),
          bijector=tfp.bijectors.Identity())

        states, kernel_results = tfp.mcmc.sample_chain(
          kernel=sampler,
          current_state=init_state,
          num_results=self.num_MCMC_steps,
          num_burnin_steps=0)


      #tf.print(self.step_size_log)
      # tf.print(tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
      # kernel_results.inner_results.inner_results.accepted_results.log_accept_ratio, 0.))))
      # tf.print(tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(
      # kernel_results.log_accept_ratio, 0.))))

      #tf.print(kernel_results)

      


    
      return states, kernel_results


  def reparameterize(self, mean_z, logvar_z, target_log_prob_fn, x):
     #samples from final variational distribution via reparam trick
     #returns final sample along with MCMC kernel results
    eps = tf.random.normal(shape = mean_z.shape)
    init_state = self.reparameterize_initial_sample(mean_z, logvar_z, eps)
    if self.num_MCMC_steps>0:
      z_trace, kernel_results = self.apply_MCMC_kernels(init_state, target_log_prob_fn, x)
      z_final = z_trace[-1]
    else:
      z_final=init_state
      kernel_results = None
    return z_final, kernel_results

  def decode(self, z, apply_sigmoid=False):
    if self.likelihood == 'Bernoulli':
      logits = self.decoder(z)
      if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
      return logits

    elif self.likelihood == 'Normal' or self.likelihood == 'Log_Normal':
      #x_mean, x_std = tf.split(self.decoder(z), 2, axis=-1)
      x_mean = self.decoder(z)
      #log_var_x = tf.clip_by_value(self.observation_log_var, clip_value_min = -7.0, clip_value_max = 10.)
      log_var_x = self.observation_log_var

      return x_mean, log_var_x

    elif self.likelihood == 'logistic_mix':
      
      params = self.decoder(z)
      
      return params

    elif self.likelihood == 'logistic':
      # mean, log_var =  tf.split(self.decoder(z), num_or_size_splits=2, axis=-1)
      # var = tf.math.exp(log_var)
      x_mean = self.decoder(z)
      log_var_x = self.observation_log_var

      return x_mean, log_var_x

    elif self.likelihood == 'Categorical':
      logits = self.decoder(z)

      return logits



  def synthesize_data(self, samples,mode = 'mean'):

    # if self.prior == 'Vamp_prior':
    # prior_mean, prior_log_var = self.encode(self.pseudo_inputs(None))
    # z = self.recompute_prior(prior_mean=prior_mean, prior_log_var=prior_log_var,samples = samples)

    #if self.prior == 'Gaussian_Mixture':
    # prior_mean, prior_log_var = self.prior_mean, self.prior_log_var
    # z = self.recompute_prior(prior_mean=prior_mean, prior_log_var=prior_log_var,samples = samples)

    # else:
    z = self.prior_dist.sample(samples)

    if self.likelihood == 'logistic_mix':
      params = self.decode(z)

      predictions = sample_from_discretized_mix_logistic(l = params, nr_mix = 10)

      

    elif self.likelihood == 'Normal' or self.likelihood == 'logistic':

      predictions, __ = self.decode(z)
      #predictions = tfd.Normal(predictions, tf.math.exp(0.5 * self.observation_log_var)).sample()


    elif self.likelihood == 'Bernoulli':

      logits = self.decode(z, False)


      if mode == 'mean':
        predictions = tfd.Bernoulli(logits = logits).mean()
      if mode == 'mode':
        predictions = tfd.Bernoulli(logits = logits).mode()
      if mode == 'random_sample':
        predictions = tfd.Bernoulli(logits = logits).sample()

      #tf.print(predictions)


    elif self.likelihood == 'Categorical':

      predictions = self.decode(z)
      predictions = tfd.Categorical(logits = predictions).sample()
      predictions  = tf.expand_dims(predictions, axis = -1)
      #predictions = tf.nn.softmax(predictions)

    # mean_z, logvar_z = self.encode(self.pseudo_inputs)

    # z = tfd.MultivariateNormalDiag(mean_z, tf.math.exp(logvar_z)).sample(samples)

    

    #sample = tfd.Normal(x_hat, tf.math.exp(x_var)).sample()

    return predictions



  def marginal_log_likelihood_estimate(self, x_test, num_particles = 500, scaling = 1.5, mcmc = False):
    mean_z, logvar_z = self.encode(x_test)

    # We consider first as proposal densities the initial variational distribution as a proposal
    ## with variance increased by a scaling factor
    logvar_z_scaled = logvar_z + tf.math.log(scaling)
    eps = tf.random.normal(shape = [num_particles] + mean_z.shape)

    expanded_x_test = tf.tile(x_test[tf.newaxis, :], [num_particles] + [1 for i in range(len(x_test.shape))])
    #expanded_x_test = tf.repeat(tf.expand_dims(x_test, axis = 0), repeats = num_particles, axis = 0)

    # Compute mean and variance for the initial state
    if mcmc == False:
      

      # Sample initial z from the variational density
      init_state = self.reparameterize_initial_sample(mean_z, logvar_z_scaled, eps)

      # density of the proposal
      log_qz_x_1 = log_normal_pdf(init_state, mean_z, logvar_z_scaled, raxis = -1)
      particle_llh = self.llh(z = tf.reshape(init_state, [-1, init_state.shape[-1]]),x = tf.reshape(expanded_x_test, [-1] + x_test.shape[1:].as_list()))
      particle_llh = tf.reshape(particle_llh, [num_particles, -1])

      # if self.prior == 'Vamp_prior':
      # pseudo_mean, pseudo_logvar = self.encode(self.pseudo_inputs(None))
      # logpz = self.recompute_prior(prior_mean=pseudo_mean, prior_log_var=pseudo_logvar,z=init_state)
      # ratios_proposal = logpz + particle_llh - log_qz_x_1

      # if self.prior == 'Gaussian_Mixture':
      # pseudo_mean, pseudo_logvar = self.prior_mean, self.prior_log_var
      # logpz = self.recompute_prior(prior_mean=pseudo_mean, prior_log_var=pseudo_logvar,z=init_state)
      # ratios_proposal = logpz + particle_llh - log_qz_x_1

      #   ratios_proposal = logpz + particle_llh - log_qz_x_1
      #else:
      ratios_proposal = self.prior_dist.log_prob(init_state) + particle_llh - log_qz_x_1


      #ratios_proposal = self.prior_dist(z = init_state[0], pseudo_inputs = self.pseudo_inputs.call(), C = 100) + particle_llh - log_qz_x_1

      #print effective sample size of the particles
      log_normalised_weights = ratios_proposal - (tf.reduce_logsumexp(ratios_proposal, axis = 0))
      ess_1 = tf.reduce_mean(1./tf.math.exp(tf.reduce_logsumexp(2.*log_normalised_weights, axis = 0)))
      # tf.print('ess proposal 1', ess_1)

      # importance sample estimate
      log_p_x_estimate = tf.reduce_mean(-tf.math.log(tf.cast(num_particles, tf.float32)) + tf.math.reduce_logsumexp(ratios_proposal, axis = 0))

      # second proposal using the scaled variance of the encoder around samples from the Markov chain
      if self.likelihood == 'logistic':
        log_p_x_estimate = log_p_x_estimate / np.prod(self.data_dim) / np.log(2)
      
      return log_p_x_estimate, ess_1

    if mcmc == True:
      #tf.print('mcmc_sampling')

      x = tf.reshape(expanded_x_test, [-1] + x_test.shape[1:].as_list())
      #x = tf.repeat(x_test, repeats = num_particles, axis = 0)
      #print('post x',x.shape)
      z0 = tf.reshape(self.reparameterize_initial_sample(mean_z, logvar_z, eps), [-1, eps.shape[-1]])

      if self.sampler == 'gradMALA' or self.sampler == 'gradHMC':

        pre_cond_operator = self.pre_cond_fn(x)

        momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
            loc = tf.zeros_like(z0),
            precision_factor = pre_cond_operator
          )

        learned_precond_hmc = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
          target_log_prob_fn = self.target_log_prob_fn(x),
          step_size = 1.,
          num_leapfrog_steps = self.num_leapfrog_steps,
          momentum_distribution = momentum_distribution
        )

        #@tf.function
        def do_sampling():
          return tfp.mcmc.sample_chain(
          kernel=learned_precond_hmc,
          current_state=z0,
          num_results=self.num_MCMC_steps,
          num_burnin_steps=0,
          trace_fn=None)
      
        z_trace = do_sampling()

      else:
        z_trace, _ = self.apply_MCMC_kernels(z0, self.target_log_prob_fn(x), x, training = False)
      proposal_dist2 = tfd.MultivariateNormalDiag(loc = tf.reshape(z_trace[-1], eps.shape),scale_diag = tf.math.exp(.5*logvar_z_scaled))
      init_state_mcmc = proposal_dist2.sample()
      particle_llh2 = self.llh(z = tf.reshape(init_state_mcmc, [-1, eps.shape[-1]]), x = x)
      particle_llh2 = tf.reshape(particle_llh2, [num_particles, -1])
      ratios_proposal2 = self.prior_dist.log_prob(init_state_mcmc) + particle_llh2 - proposal_dist2.log_prob(init_state_mcmc)
      log_p_x_estimate2 = tf.reduce_mean(-tf.math.log(tf.cast(num_particles, tf.float32)) + tf.math.reduce_logsumexp(ratios_proposal2, axis = 0))
      log_normalised_weights2 = ratios_proposal2 - (tf.reduce_logsumexp(ratios_proposal2, axis = 0))
      ess_2 = tf.reduce_mean(1. / tf.math.exp(tf.reduce_logsumexp(2. * log_normalised_weights2, axis = 0)))
      

      # proposal_dist3 = tfd.MultivariateNormalDiag(loc = tf.reshape(z_trace[-1], eps.shape),scale_diag = tf.reshape(scaling * tfp.stats.stddev(z_trace, 0),eps.shape))
      # init_state_mcmc3 = proposal_dist3.sample()
      # particle_llh3 = self.llh(z = tf.reshape(init_state_mcmc3, [-1, eps.shape[-1]]), x = x)
      # particle_llh3 = tf.reshape(particle_llh3, [num_particles, -1])
      # ratios_proposal3 = self.prior_dist.log_prob(init_state_mcmc3) + particle_llh3 - proposal_dist3.log_prob(init_state_mcmc3)
      # log_p_x_estimate3 = tf.reduce_mean(-tf.math.log(tf.cast(num_particles, tf.float32)) + tf.math.reduce_logsumexp(ratios_proposal3, axis = 0))
      # log_normalised_weights3 = ratios_proposal3 - (tf.reduce_logsumexp(ratios_proposal3, axis = 0))
      # ess_3 = tf.reduce_mean(1. / tf.math.exp(tf.reduce_logsumexp(2. * log_normalised_weights3, axis = 0)))

      # tf.print('ess proposal 2', ess_2)
      #tf.print('ess proposal 3', ess_3)
      if self.likelihood == 'logistic':
        log_p_x_estimate2 = log_p_x_estimate2 / np.prod(self.data_dim) / np.log(2)

    #tf.print(log_p_x_estimate,log_p_x_estimate2,log_p_x_estimate3)


    return log_p_x_estimate2, ess_2



  def marginal_loglikelihood(self, x, samples = 512):

    mean_z, logvar_z = self.encode(x)
    eps = tf.random.normal(shape = [num_particles] + mean_z.shape)








    estimate = 0


    return estimate


  def annealed_importance_sampling_estimate(self, x, num_parallel_chains = 1, num_particles = 5000):


    #proposal = tfd.Normal(loc=tf.zeros(num_parallel_chains), scale=tf.ones(num_parallel_chains))
    # model = ais.HAIS(qz=prior, log_likelihood_fn=unnormalized_log_gamma_lpdf)

    # mean_z, logvar_z = self.encode(x)
    # eps = tf.random.normal(shape = mean_z.shape)

    # init_state = self.reparameterize_initial_sample(mean_z, logvar_z, eps)
    # target_log_prob_fn = self.target_log_prob_fn(x)
    # proposal = tfd.MultivariateNormalDiag(loc=mean_z,scale_diag = tf.math.exp(0.5*logvar_z))


    # model = ais.HAIS(proposal=proposal, log_target=target_log_prob_fn, stepsize=.7, adapt_stepsize=False)

    # # Set up an annealing schedule
    # schedule = ais.get_schedule(T=1000, r=4)

    # # Set up the computation graph
    # logw, z_i, eps, avg_acceptance_rate = model.ais(schedule)
    # log_normalizer = model.log_normalizer(logw, samples_axis=0)




    mean_z, logvar_z = self.encode(x)
    #logvar_z_scaled = logvar_z + tf.math.log(scaling)
    #eps = tf.random.normal(shape = [num_parallel_chains] + mean_z.shape)
    eps = tf.random.normal(shape = mean_z.shape)
    init_state = self.reparameterize_initial_sample(mean_z, logvar_z, eps)
    #tf.print('init_sate_shape',init_state.shape)
    #init_state = tf.repeat(init_state, num_parallel_chains, axis = 0)

    #x = tf.repeat(x, repeats= num_parallel_chains, axis = 0)

    

    target_log_prob_fn = self.target_log_prob_fn(x)
    proposal = tfd.MultivariateNormalDiag(loc=mean_z,scale_diag = tf.math.exp(0.5*logvar_z))

    sampler = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=self.num_leapfrog_steps,
            step_size=1.0,store_parameters_in_results = True),
            bijector=tfp.bijectors.Identity())

      
    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
          inner_kernel=sampler,
          num_adaptation_steps=100,
          target_accept_prob=0.65
          )


    weight_samples, ais_weights, kernel_results = (
    tfp.mcmc.sample_annealed_importance_chain(
      num_steps=num_particles,
      proposal_log_prob_fn=proposal.log_prob,
      target_log_prob_fn=target_log_prob_fn,
      current_state=init_state,
      make_kernel_fn=adaptive_sampler))
      


    estimate = (tf.reduce_logsumexp(ais_weights) - np.log(num_parallel_chains))
    estimate = log_normalizer
    #tf.print('acceptance_rate', tf.reduce_mean(tf.cast(kernel_results.inner_results.is_accepted, dtype=tf.float32)))
    tf.print('logpx_estimate', estimate)

    return estimate





  # @tf.function
  # def compute_loss(self, x, debug = False):
  #   return self._compute_loss(x, debug = debug)


  # Define loss function
  def compute_loss(self, x, mcmc = False,debug = False, beta = 1.):
    mean_z, logvar_z = self.encode(x)
    
    eps = tf.random.normal(shape = mean_z.shape)
    #initial sample is from the transformed random variable
    init_state = self.reparameterize_initial_sample(mean_z, logvar_z, eps)

    # need to recompute the prior if trainable for gradient computations
    #if self.prior == 'Vamp_prior':
    # pseudo_mean, pseudo_logvar = self.encode(self.pseudo_inputs(None))
    # logpz = self.recompute_prior(prior_mean=pseudo_mean, prior_log_var=pseudo_logvar,z=init_state)

    #if self.prior == 'Gaussian_Mixture':
    # pseudo_mean, pseudo_logvar = self.prior_mean, self.prior_log_var
    # logpz = self.recompute_prior(prior_mean=pseudo_mean, prior_log_var=pseudo_logvar,z=init_state)

    # else:
    logpz = self.prior_dist.log_prob(init_state)

    #log likelihood for initial sample for training the initial variational distribution
    decoder_loss_init_sample = -self.llh(z=init_state, x=x)
    
    logqz_x = log_normal_pdf(init_state, mean_z, logvar_z)

    KL_loss = (logqz_x - logpz)

    encoder_loss = -(self.beta * (logpz - logqz_x) - decoder_loss_init_sample)

    target_log_prob_fn = self.target_log_prob_fn(x)

    # We just apply MCMC steps from the initial state which is not sampled again with a different eps
    if mcmc == True:
      #tf.print('sampling')
      z_trace, kernel_results = self.apply_MCMC_kernels(init_state,target_log_prob_fn, x)
      z_final = z_trace[-1]

    else:
      z_final = init_state

    # Train generative model using log-likelihood at the final MCMC state
    # stop gradient of final sample as the proposal/sample depends on generative parameters
    # and we neglect this here
    z_final = tf.stop_gradient(z_final)

    
    if self.prior != 'Isotropic_Gaussian':
      decoder_loss_final_sample = - self.llh(z=z_final, x=x)
      #+ self.prior_dist(z = z_final,pseudo_inputs = self.pseudo_inputs.call(), C = 1000)
    else:
      decoder_loss_final_sample = - self.llh(z=z_final, x=x)


    if debug == True:
      print('logpz', logpz)
      print('logqz_x', logqz_x)
      print('init_state', init_state)
      print('mean_z', mean_z)
      print('logvar_z', logvar_z)
      tf.print('encoder_loss', encoder_loss)
      tf.print('decoder_loss_final_sample', decoder_loss_final_sample)
      tf.print('decoder_loss_init_sample',decoder_loss_init_sample)
      tf.print('encoder_loss_shape', encoder_loss.shape)
      tf.print('decoder_loss_final_shape', decoder_loss_final_sample.shape)
      tf.print('decoder_loss_init_shape',decoder_loss_init_sample.shape)
      tf.print(self.pseudo_inputs.trainable_variables)
      tf.print('KL beta',self.beta)




    return encoder_loss, decoder_loss_final_sample, decoder_loss_init_sample, KL_loss


  @tf.function
  def train_step(self, x,mcmc = False,beta = 1.):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the self's parameters.
    """
    with tf.GradientTape(persistent = True) as tape:
      encoder_loss, decoder_loss, __, __ = self.compute_loss(x, mcmc = mcmc, beta = beta)

    
    gradients_encoder = tape.gradient(encoder_loss, self.encoder.trainable_weights)
    gradients_decoder = tape.gradient(decoder_loss, self.decoder.trainable_weights)

    
    self.optimizer_encoder.apply_gradients(zip(gradients_encoder, self.encoder.trainable_weights))
    self.optimizer_decoder.apply_gradients(zip(gradients_decoder, self.decoder.trainable_weights))


    # if prior is trainable
    # gradients_prior_mean = tape.gradient(encoder_loss, [self.prior_mean])
    # gradients_prior_log_var = tape.gradient(encoder_loss, [self.prior_log_var])
    # self.optimizer_prior.apply_gradients(zip(gradients_prior_mean, [self.prior_mean]))
    # self.optimizer_prior.apply_gradients(zip(gradients_prior_log_var, [self.prior_log_var]))

    #tf.print(self.prior_mean)
    # if self.prior != 'Isotropic_Gaussian':

    #   gradients_prior = tape.gradient(encoder_loss, self.prior_params.trainable_variables)
    #   self.optimizer_prior.apply_gradients(zip(gradients_prior, self.prior_params.trainable_variables))
    #if self.prior == 'Gaussian_Mixture':
    # gradients_prior_mean = tape.gradient(encoder_loss, [self.prior_mean])
    # gradients_prior_log_var = tape.gradient(encoder_loss, [self.prior_log_var])
    # self.optimizer_prior.apply_gradients(zip(gradients_prior_mean, [self.prior_mean]))
    # self.optimizer_prior.apply_gradients(zip(gradients_prior_log_var, [self.prior_log_var]))

    
    # gradients_pseudo_inputs = tape.gradient(encoder_loss, self.pseudo_inputs.trainable_variables)
    # self.optmizer_pseudo_inputs.apply_gradients(zip(gradients_pseudo_inputs,self.pseudo_inputs.trainable_variables))
    #tf.print(z)

    #gradients_observation_noise = tape.gradient(decoder_loss, [self.observation_log_var])
    #self.optimizer_observation_noise.apply_gradients(zip(gradients_observation_noise, [self.observation_log_var]))
    #tf.print(self.observation_log_var)
    


  @tf.function
  def compute_grads(self, x):
    """Computes gradient for testing.

    This function computes the loss and gradients, and uses the latter to
    update the self's parameters.
    """
    with tf.GradientTape(persistent = True) as tape:
      encoder_loss, decoder_loss, __ = self.compute_loss(x)
    gradients_encoder = tape.gradient(encoder_loss, self.encoder.trainable_weights)
    gradients_decoder = tape.gradient(decoder_loss, self.decoder.trainable_weights)

    return gradients_encoder, gradients_decoder


