from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
from grad_adaptive_hmc import AdaptiveHMC
import pdb


class CHVAE(tf.keras.Model):
  """hierarchical variational autoencoder class in a central paraterisation."""

  def __init__(self, encoder_layers, decoder_layers,
               top_down_deterministic_layers, bottom_up_deterministic_layers,
               observation_layer, input_dim, latent_dims, generative_deterministic_dims,
               pre_cond_fn, pre_cond_params,
               likelihood, sampler = "gradHMC",
               learning_rate = 1e-4, learning_rate_mcmc = 1e-4, num_MCMC_steps = 5,
               num_leapfrog_steps = 1, beta = 1.,
               min_entropy_weight = .001, max_entropy_weight = 100, biased_grads = True,
               use_residual = False, min_prior_scale = 1e-1, train_prior = True):

    super(CHVAE, self).__init__()

    self.encoder_layers = encoder_layers
    self.decoder_layers = decoder_layers
    self.top_down_deterministic_layers = top_down_deterministic_layers
    self.bottom_up_deterministic_layers = bottom_up_deterministic_layers
    self.observation_layer = observation_layer
    self.input_dim = input_dim
    self.latent_dims = latent_dims
    self.generative_deterministic_dims = generative_deterministic_dims
    self.pre_cond_fn = pre_cond_fn
    self.pre_cond_params = pre_cond_params
    self.num_leapfrog_steps = num_leapfrog_steps
    self.optimizer_entropy_weight = tf.keras.optimizers.Adam(10 * learning_rate)
    self.optimizer_mcmc = tf.keras.optimizers.Adam(learning_rate_mcmc)
    self.num_MCMC_steps = num_MCMC_steps
    self.proposal_entropy_weight = tf.Variable(initial_value = 1., trainable = True)
    self.penalty_param = tf.Variable(initial_value = 1000., trainable = True)
    self.min_entropy_weight = min_entropy_weight
    self.max_entropy_weight = max_entropy_weight
    self.learning_rate = learning_rate
    self.sampler = sampler
    self.likelihood = likelihood
    self.biased_grads = biased_grads
    self.use_residual = use_residual
    self.min_prior_scale = min_prior_scale
    self.observation_log_var = tf.Variable(initial_value = -2., trainable = True)
    self.optimizer_observation_noise = tf.keras.optimizers.Adam(learning_rate)
    self.train_prior = train_prior

    # to build deterministic layers, sample once
    self.sample_conditional(tf.zeros([1, input_dim]))

    self.encoder_params = [f.trainable_weights for f in self.encoder_layers] + [
      f.trainable_weights for f in self.bottom_up_deterministic_layers]
    if self.train_prior:
      self.decoder_params = [f.trainable_weights for f in self.decoder_layers] + [
        f.trainable_weights for f in self.top_down_deterministic_layers] + [self.observation_layer.trainable_variables]
    else:
      self.decoder_params = [self.observation_layer.trainable_variables]

    #print(self.encoder_params)
    #print(self.decoder_params)
    self.optimizer_decoder = [tf.keras.optimizers.Adam(learning_rate) for _ in self.decoder_params]
    self.optimizer_encoder = [tf.keras.optimizers.Adam(learning_rate) for _ in self.encoder_params]

    # self.optimizer_encoder_mean = tf.keras.optimizers.Adam(learning_rate)
    # self.optimizer_encoder_var = tf.keras.optimizers.Adam(learning_rate * 0.5)
    self.beta = beta

  # log target function
  def target_log_prob_fn(self, x):
    def fn(z):
      prior_log_probs, d_p, _ = self.prior_log_prob(z)
      prior_log_probs = tf.reduce_sum(prior_log_probs, -1)
      return prior_log_probs + self.llh(z, x)

    return fn
    # return lambda z: tf.reduce_sum(self.prior_log_prob(z), 0) + self.llh(z, x)

  def observation_model(self, z):
    d_ps = []  # deterministic variables
    dims = self.latent_dims
    z_top_down_path = []

    for i, f in enumerate(self.top_down_deterministic_layers):
      if i == 0:
        d_p = f(tf.zeros([z.shape[0], 1]))
      else:
        if self.use_residual:
          d_p = f(tf.concat([z_top, d_p], axis = -1))
        else:
          d_p = f(z_top)
      d_ps += [d_p]
      p_mean_top_down, p_log_var_top_down = tf.split(self.decoder_layers[i](d_p),num_or_size_splits = 2, axis = 1)

      z_top = tf.gather(z, tf.range(sum(dims[:i]), sum(dims[:i + 1])), axis = -1)
      z_top_down_path += [z_top]

    z_top_down_path = tf.concat(z_top_down_path, -1)



    if not self.use_residual:
      return self.observation_layer(tf.gather(z_top_down_path, tf.range(sum(self.latent_dims[:-1]), sum(self.latent_dims)), axis = -1))
    else:
      return self.observation_layer(tf.concat([tf.gather(z_top_down_path, tf.range(sum(self.latent_dims[:-1]), sum(self.latent_dims)), axis = -1),d_p], axis=-1))

  def llh(self, z, x):

    if self.likelihood == 'Bernoulli':
      # TODO check
      # pdb.set_trace()
      if self.use_residual:
        x_logit = self.observation_model(z)
      else:
        x_logit = self.observation_model(z)

      decoder_llh = tf.reduce_sum(tfd.Bernoulli(logits = x_logit).log_prob(x), -1)

    elif self.likelihood == 'Normal':

      if self.use_residual:
        # mean, log_var = self.observation_model(z), self.observation_log_var
        # linear case
        mean, log_var = tf.split(self.observation_model(z), num_or_size_splits = 2, axis = 1)
      else:
        # mean, log_var = self.observation_model(z), self.observation_log_var
        # linear case
        mean, log_var = tf.split(self.observation_model(z), num_or_size_splits = 2, axis = 1)

      decoder_llh = tf.reduce_sum(tfd.Normal(loc = mean, scale = 1e-4 + tf.math.exp(.5 * log_var)).log_prob(x), -1)
    return decoder_llh

  def prior_log_prob(self, z):

    log_prob_top_down_path = []  # log-prob of stochastic latents
    d_ps = []  # deterministic variables
    zs = []
    dims = self.latent_dims
    for i, f in enumerate(self.top_down_deterministic_layers):
      if i == 0:
        d_p = f(tf.zeros([z.shape[0], 1]))
      else:
        if self.use_residual:
          d_p = f(tf.concat([z_top, d_p], axis = -1))
        else:
          d_p = f(z_top)
      d_ps += [d_p]
      p_mean_top_down, p_log_var_top_down = tf.split(self.decoder_layers[i](d_p),num_or_size_splits = 2, axis = 1)


      z_top = tf.gather(z, tf.range(sum(dims[:i]), sum(dims[:i + 1])), axis = -1)
      z_log_prob = tfd.MultivariateNormalDiag(loc = p_mean_top_down, scale_diag = self.min_prior_scale + tf.math.exp(.5 * (p_log_var_top_down))).log_prob(z_top)
      log_prob_top_down_path += [z_log_prob]
      zs += [z_top]

    return tf.stack(log_prob_top_down_path, -1), d_ps, zs

  def deterministic_bottom_up(self, x):

    deterministic_path_bottom_up = []
    d_bu = x
    for i, f in enumerate(self.bottom_up_deterministic_layers):
      d_bu = f(d_bu)
      deterministic_path_bottom_up += [d_bu]
    return deterministic_path_bottom_up

  def sample_prior(self, batch_size):
    d_ps = []
    z_top_down_path = []
    eps_top_down_path = []
    log_prob_top_down_path = []
    for i, f in enumerate(self.top_down_deterministic_layers):
      if i == 0:
        # d_p = tf.zeros([x.shape[0], self.generative_deterministic_dims[0]])
        d_p = f(tf.zeros([batch_size, 1]))
      else:
        if self.use_residual:
          d_p = f(tf.concat([z_top, d_p], axis = -1))
        else:
          d_p = f(z_top)
      d_ps += [d_p]

      p_mean_top_down, p_log_var_top_down = tf.split(self.decoder_layers[i](d_p),num_or_size_splits = 2, axis = 1)

      z_top = tfd.MultivariateNormalDiag(loc = p_mean_top_down, scale_diag = self.min_prior_scale + tf.math.exp(.5 * (p_log_var_top_down))).sample()
      log_prob_top_down_path += [tfd.MultivariateNormalDiag(loc = p_mean_top_down, scale_diag = self.min_prior_scale + tf.math.exp(.5 * (p_log_var_top_down))).log_prob(z_top)]
      z_top_down_path += [z_top]

    return tf.concat(z_top_down_path, -1), tf.stack(log_prob_top_down_path, 0), tf.concat(z_top_down_path, -1), d_ps

  def sample_conditional(self, x):

    deterministic_path_bottom_up = self.deterministic_bottom_up(x)
    deterministic_path_bottom_up_reordered = deterministic_path_bottom_up[::-1]
    z_top_down_path = []
    eps_top_down_path = []
    log_prob_top_down_path = []
    d_ps = []
    KL_div = 0

    for i, f in enumerate(self.top_down_deterministic_layers):
      if i == 0:
        d_p = f(tf.zeros([x.shape[0], 1]))
      else:
        if self.use_residual:
          d_p = f(tf.concat([z_top, d_p], axis = -1))
        else:
          d_p = f(z_top)

      d_ps += [d_p]
      d_bu = deterministic_path_bottom_up_reordered[i]
      q_mean_top_down, q_log_var_top_down = tf.split(self.encoder_layers[i](tf.concat([d_bu, d_p], -1)),
                                                     num_or_size_splits = 2, axis = 1)
      # tf.print('layer',i)
      # tf.print('q_mean_top_down',q_mean_top_down)
      # tf.print('q_log_var_top_down',q_log_var_top_down)


      p_mean_top_down, p_log_var_top_down = tf.split(self.decoder_layers[i](d_p),
                                                     num_or_size_splits = 2, axis = 1)

      # tf.print('max_q_mean_top_down',tf.math.reduce_max(q_mean_top_down))
      # tf.print('min_q_mean_top_down',tf.math.reduce_min(q_mean_top_down))
      # tf.print('max_q_log_var_top_down',tf.math.reduce_max(q_log_var_top_down))
      # tf.print('min_q_log_var_top_down',tf.math.reduce_min(q_log_var_top_down))
      # tf.print('max_p_mean_top_down',tf.math.reduce_max(p_mean_top_down))
      # tf.print('min_p_mean_top_down',tf.math.reduce_min(p_mean_top_down))
      # tf.print('max_p_log_var_top_down',tf.math.reduce_max(p_log_var_top_down))
      # tf.print('min_p_log_var_top_down',tf.math.reduce_min(p_log_var_top_down))




    
      q_z_top_down_loc =  p_mean_top_down + tf.math.exp(.5 * (p_log_var_top_down)) * q_mean_top_down
      q_z_top_down_scale =  self.min_prior_scale + tf.math.exp(.5 * (p_log_var_top_down)) * tf.math.exp(.5 * (q_log_var_top_down))
      q_z = tfd.MultivariateNormalDiag(loc = q_z_top_down_loc, scale_diag = q_z_top_down_scale)

      z_top = q_z.sample()
      z_log_prob = q_z.log_prob(z_top)
      z_top_down_path += [z_top]
      log_prob_top_down_path += [z_log_prob]
      # tf.print('layer',i)
      KL_div += tfp.distributions.kl_divergence(q_z,tfd.MultivariateNormalDiag(loc = p_mean_top_down, scale_diag = self.min_prior_scale + tf.math.exp(0.5 * p_log_var_top_down)))
      # tf.print('KL_div',tf.reduce_sum(KL_div))




      

      # tf.print('d_p',d_p)
      # tf.print('d_bu', d_bu)
      # tf.print('q_mean_top_down',q_mean_top_down)
      # tf.print('q_log_var_top_down',q_log_var_top_down)
      # tf.print('p_mean_top_down',p_mean_top_down)
      # tf.print('p_log_var_top_down',p_log_var_top_down)



    return tf.concat(z_top_down_path, -1), tf.stack(log_prob_top_down_path, -1), tf.concat(z_top_down_path, -1), d_ps, KL_div



  def apply_MCMC_kernels(self, init_state, target_log_prob_fn, x):
    # apply MCMC kernels kernels

    sampler = self.sampler
    if sampler == "gradHMC":
      kernel = AdaptiveHMC(
        num_leapfrog_steps = self.num_leapfrog_steps,
        target_log_prob_fn = target_log_prob_fn,
        pre_cond_fn = self.pre_cond_fn,
        pre_cond_params = self.pre_cond_params,
        x = x,
        learning_rate_beta = self.learning_rate,
        proposal_entropy_weight = self.proposal_entropy_weight,
        penalty_param = self.penalty_param,
        min_entropy_weight = self.min_entropy_weight,
        max_entropy_weight = self.max_entropy_weight,
        biased_grads = self.biased_grads,
        optimizer = self.optimizer_mcmc,
      )

      states, kernel_results = tfp.mcmc.sample_chain(
        num_results = self.num_MCMC_steps,
        current_state = init_state,
        previous_kernel_results = kernel.bootstrap_results(init_state = init_state),
        kernel = kernel,
        num_burnin_steps = 0)

      return states, kernel_results

    if sampler == 'dsHMC':
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

      return states, kernel_results

  def sample_conditional_MCMC(self, x):

    z_init, log_qzs, zs, dp, __ = self.sample_conditional(x)

    if self.num_MCMC_steps > 0:
      z_trace, kernel_results = self.apply_MCMC_kernels(z_init, self.target_log_prob_fn(x), x)
      z_final = z_trace[-1]
    else:
      z_final = z_init
      kernel_results = None

    return z_final, kernel_results

  @tf.function
  def mean_reconstruced_image(self, x):
    z, kernel_results = self.sample_conditional_MCMC(x)
    _, d_p,_ = self.prior_log_prob(z)

    if self.likelihood == 'Bernoulli':
      x_logit = self.observation_model(z)
      return tf.sigmoid(x_logit)
    elif self.likelihood == 'Normal':
      if self.use_residual:
        mean, log_var = self.observation_model(z), self.observation_log_var
      else:
        mean, log_var = self.observation_model(z), self.observation_log_var
    return mean

  @tf.function
  def mean_prior_image(self, batch_size):
    z, _, z, d_p = self.sample_prior(batch_size)
    # _, d_p_init = self.prior_log_prob(z_init)
    if self.likelihood == 'Bernoulli':
      x_logit = self.observation_model(z)
      return tf.sigmoid(x_logit)

    elif self.likelihood == 'Normal':
      if self.use_residual:
        mean, log_var = self.observation_model(z), self.observation_log_var
      else:
        mean, log_var = self.observation_model(z), self.observation_log_var
    return mean

  def synthesize_data(self, samples):
    z, _, z, d_p = self.sample_prior(samples)
    # _, d_p_init = self.prior_log_prob(z_init)
    if self.likelihood == 'Bernoulli':
      x_logit = self.observation_model(z)
      x_logit = tf.sigmoid(x_logit)
      x_logit = tf.reshape(x_logit, (samples, 28,28,1))
      return tf.convert_to_tensor(x_logit)

    elif self.likelihood == 'Normal':
      if self.use_residual:
        mean, log_var = self.observation_model(z), self.observation_log_var
      else:
        mean, log_var = self.observation_model(z), self.observation_log_var
    return mean


  def compute_loss(self, x, debug = False):
    tf.print(self.num_MCMC_steps)
    return self._compute_loss(x, debug = debug)

  # Define loss function
  def compute_loss(self, x):

    z_init, log_qzs_init, zs_init, d_p_init, kl_div = self.sample_conditional(x)

    log_qzs_init_sum = tf.reduce_sum(log_qzs_init, -1)
    log_pzs_init, d_p_init_, _ = self.prior_log_prob(z_init)
    log_pzs_init_sum = tf.reduce_sum(log_pzs_init, -1)
    decoder_loss_init_sample = self.llh(z = z_init, x = x)
    # tf.debugging.assert_equal(tf.concat(d_p_init, 1), tf.concat(d_p_init_, 1))
    # standard ELBO loss (based on KL) for training initial variational distribution

    # tf.print('log_pzs_init', log_pzs_init)
    # tf.print('log_qzs_init', log_qzs_init)
    # tf.print('KL divergance per layer', log_qzs_init - log_pzs_init)
    # tf.print('decoder_loss_init_sample', decoder_loss_init_sample)




    # tf.print('log_qzs_init_sum',log_qzs_init_sum)
    # tf.print('z_init',z_init)
    # tf.print('log_qzs_init',log_qzs_init)
    # tf.print('zs_init',zs_init)
    # tf.print('d_p_init',d_p_init)

    # tf.print('kl_div', kl_div)
    # tf.print('decoder_loss_init_sample',tf.reduce_sum(decoder_loss_init_sample))

    # encoder_loss = self.beta*(log_qzs_init_sum - log_pzs_init_sum) + decoder_loss_init_sample
    encoder_loss = self.beta*kl_div - decoder_loss_init_sample
    # tf.print('beta',self.beta)

    # use MCMC sample as posterior approximation if MCMC steps are used, otherwise keep initial values
    if self.num_MCMC_steps > 0:
      zs_trace, kernel_results = self.apply_MCMC_kernels(z_init, self.target_log_prob_fn(x), x)
      z_final = zs_trace[-1]
      z_final = tf.stop_gradient(z_final)
      log_pzs_final, d_p_final, _ = self.prior_log_prob(z_final)
      d_p_final = [tf.stop_gradient(y) for y in d_p_final]

    else:
      z_final = z_init
      log_pzs_final = log_pzs_init
      d_p_final = d_p_init

    log_pzs_final_sum = tf.reduce_sum(log_pzs_final, -1)

    decoder_loss_final_sample = - self.llh(z = z_final, x = x) - log_pzs_final_sum

    # tf.print('encoder_loss',tf.reduce_sum(encoder_loss))
    # tf.print('decoder_loss_final_sample',tf.reduce_sum(decoder_loss_final_sample))

    return encoder_loss, decoder_loss_final_sample, decoder_loss_init_sample

  @tf.function
  def train_step(self, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the self's parameters.
    """


    with tf.GradientTape(persistent = True) as tape:
      tape.watch(x)
      encoder_loss, decoder_loss, decoder_loss_init = self.compute_loss(x)


    gradients_encoder = tape.gradient(encoder_loss, self.encoder_params)
    gradients_decoder = tape.gradient(decoder_loss, self.decoder_params)
    gradients_decoder_init = tape.gradient(decoder_loss_init, self.decoder_params)
    # print(gradients_encoder)
    #print(gradients_decoder)
    # tf.print('encoder_loss',tf.reduce_sum(encoder_loss))
    # tf.print('decoder_loss_final_sample',tf.reduce_sum(decoder_loss))

    # tf.print('gradients_encoder',gradients_encoder)
    # tf.print('gradients_decoder',gradients_decoder)

    CLIP_VALUE = 10000.
    gradients_decoder = tf.nest.map_structure(
      lambda g: tf.where(tf.math.is_nan(g), tf.zeros_like(g), g),
      gradients_decoder)
    gradients_encoder = tf.nest.map_structure(
      lambda g: tf.where(tf.math.is_nan(g), tf.zeros_like(g), g),
      gradients_encoder)
    gradients_decoder = tf.nest.map_structure(
      lambda g: tf.clip_by_value(g, -CLIP_VALUE, CLIP_VALUE),
      gradients_decoder)
    gradients_encoder = tf.nest.map_structure(
      lambda g: tf.clip_by_value(g, -CLIP_VALUE, CLIP_VALUE),
      gradients_encoder)

    [opt.apply_gradients(zip(g, p)) for opt, g, p in zip(
      self.optimizer_encoder, gradients_encoder, self.encoder_params)]
    [opt.apply_gradients(zip(g, p)) for opt, g, p in zip(
      self.optimizer_decoder, gradients_decoder, self.decoder_params)]

    # self.beta = min((epoch+1)/1000, 1.)
    self.beta = 1.0


    return gradients_encoder, gradients_decoder

    #
    # self.optimizer_decoder.apply_gradients(zip(gradients_decoder, self.decoder_params))
    # self.optimizer_encoder.apply_gradients(zip(gradients_encoder, self.encoder_params))

    # TO DO: combine grad updates
    # for l in range(len(self.encoder_list)): self.optimizer_encoder[l].apply_gradients(
    #   zip(gradients_encoder[l], self.encoder_list[l].trainable_weights))
    #
    #
    # for ll in range(len(self.decoder_list)-1):
    #   self.optimizer_decoder[ll].apply_gradients(
    #   zip(gradients_encoder[1+ll+l], self.decoder_list[ll].trainable_weights))
    # self.optimizer_decoder[-1].apply_gradients(
    #   zip(gradients_decoder[-1], self.decoder_list[-1].trainable_weights))