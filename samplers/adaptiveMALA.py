from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc import transformed_kernel
import numpy as np
from tensorflow_probability.python.mcmc.internal import util as mcmc_util



#####
# code for transforming the log probability as in tfp transformed_kernel.py
# with the exception that gradients are stopped after applying the bijector
# to the transformed state
# these might be used for computing gradients for the potential energy terms, but not
# for computing the drift in the MALA proposal

def make_transformed_log_prob_stop_grads(
  log_prob_fn, bijector, direction, enable_bijector_caching = True):
  """Transforms a log_prob function using bijectors.

  Note: `direction = 'inverse'` corresponds to the transformation calculation
  done in `tfp.distributions.TransformedDistribution.log_prob`.

  Args:
    log_prob_fn: Python `callable` taking an argument for each state part which
      returns a `Tensor` representing the joint `log` probability of those state
      parts.
    bijector: `tfp.bijectors.Bijector`-like instance (or list thereof)
      corresponding to each state part. When `direction = 'forward'` the
      `Bijector`-like instance must possess members `forward` and
      `forward_log_det_jacobian` (and corresponding when
      `direction = 'inverse'`).
    direction: Python `str` being either `'forward'` or `'inverse'` which
      indicates the nature of the bijector transformation applied to each state
      part.
    enable_bijector_caching: Python `bool` indicating if `Bijector` caching
      should be invalidated.
      Default value: `True`.

  Returns:
    transformed_log_prob_fn: Python `callable` which takes an argument for each
      transformed state part and returns a `Tensor` representing the joint `log`
      probability of the transformed state parts.
  """
  if direction not in {'forward', 'inverse'}:
    raise ValueError('Argument `direction` must be either `"forward"` or '
                     '`"inverse"`; saw "{}".'.format(direction))
  fn = transformed_kernel.make_transform_fn(bijector, direction)
  ldj_fn = transformed_kernel.make_log_det_jacobian_fn(bijector, direction)

  def transformed_log_prob_fn(*state_parts):
    """Log prob of the transformed state."""
    if not enable_bijector_caching:
      state_parts = [tf.identity(sp) for sp in state_parts]
    tlp = log_prob_fn(tf.stop_gradient(*fn(state_parts)))
    tlp_rank = transformed_kernel.prefer_static.rank(tlp)
    event_ndims = [(transformed_kernel.prefer_static.rank(sp) - tlp_rank) for sp in state_parts]
    return tlp + sum(ldj_fn(state_parts, event_ndims))

  return transformed_log_prob_fn


class AdaptiveMALA(transformed_kernel.TransformedTransitionKernel):

  def __init__(self, target_log_prob_fn, bijector_fn, bijector_params, x,
               learning_rate_beta, proposal_entropy_weight, optimizer_transformation,
               opt_acceptance_rate=.5, min_entropy_weight=.001, max_entropy_weight=100., name=None):
    inner_kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
      target_log_prob_fn = target_log_prob_fn,
      step_size = 1.
    )
    self.bijector_fn = bijector_fn
    self.bijector_params = bijector_params#bijector.trainable_variables
    self.x = x
    #self.optimizer_entropy_weight = optimizer_entropy_weight
    self.optimizer_transformation = optimizer_transformation
    self.proposal_entropy_weight = proposal_entropy_weight
    self.opt_acceptance_rate = opt_acceptance_rate
    self.learning_rate_beta = learning_rate_beta
    self.min_entropy_weight = min_entropy_weight
    self.max_entropy_weight = max_entropy_weight
    self.target_log_prob_fn = target_log_prob_fn
    super().__init__(inner_kernel, bijector_fn(x), name)



  def one_step(self, current_state, previous_kernel_results, seed = None):
    with tf.GradientTape(persistent = True) as t1:

      t1.watch(self.bijector_params)
      t1.watch(self.x)
      bijector = self.bijector_fn(self.x)
      current_transformed_state = previous_kernel_results.transformed_state

      __, current_log_target_grads = tfp.math.value_and_gradient(
        self.target_log_prob_fn, tf.stop_gradient(current_state))
      with tf.GradientTape(persistent = True) as t2:
        x = tf.stop_gradient(current_transformed_state)
        t2.watch(x)
        rep_current_state = bijector(x)
      rep_transformed_current_log_target_grads = t2.gradient(rep_current_state, x,
                                                             tf.stop_gradient(current_log_target_grads))
      rep_transformed_current_log_target_grads = list(rep_transformed_current_log_target_grads) if mcmc_util.is_list_like(
        rep_transformed_current_log_target_grads) else [
        rep_transformed_current_log_target_grads]

      #TODO: rewrite without one_step method of base class
      super().__init__(self.inner_kernel, bijector)
      updated_accept_results = previous_kernel_results.inner_results.accepted_results._replace(
        diffusion_drift=tf.nest.map_structure(lambda g: .5 * self.inner_kernel.step_size**2 * g,
                                              rep_transformed_current_log_target_grads),
        grads_target_log_prob = rep_transformed_current_log_target_grads,
        target_log_prob = self._inner_kernel.target_log_prob_fn(current_transformed_state)
      )
      updated_inner_results = previous_kernel_results.inner_results._replace(
        accepted_results = updated_accept_results
      )
      previous_kernel_results = previous_kernel_results._replace(
        inner_results = updated_inner_results
      )

      next_state, kernel_results = super().one_step(current_state, previous_kernel_results, seed)
      log_accept_ratio = kernel_results.inner_results.log_accept_ratio
      proposal_log_prob = - self.bijector.forward_log_det_jacobian(kernel_results.transformed_state, 1)
      speed_measure_loss = - log_accept_ratio + self.proposal_entropy_weight * proposal_log_prob

      #TO DO:test
      #biased log-accept gradient estimates


      proposed_transformed_target_log_prob = kernel_results.inner_results.proposed_results.target_log_prob
      current_transformed_target_log_prob = previous_kernel_results.inner_results.accepted_results.target_log_prob

      proposed_transformed_state = kernel_results.inner_results.proposed_state
      _ , proposed_log_target_grads = tfp.math.value_and_gradient(
        self.target_log_prob_fn, tf.stop_gradient(bijector(proposed_transformed_state)))

      with tf.GradientTape(persistent = True) as t2:
        x = tf.stop_gradient(proposed_transformed_state)
        #x = (proposed_transformed_state)
        t2.watch(x)
        rep_proposed_state = bijector(x)
      rep_transformed_proposed_log_target_grads = t2.gradient(rep_proposed_state, x,
                                                              tf.stop_gradient(proposed_log_target_grads))
      rep_transformed_proposed_log_target_grads = list(rep_transformed_proposed_log_target_grads) if mcmc_util.is_list_like(
        rep_transformed_proposed_log_target_grads) else [
        rep_transformed_proposed_log_target_grads]

      #recover eps noise of MALA proposal on transformed space
      eps = tf.nest.map_structure(
        lambda q0,q1,g: tf.stop_gradient(1./tf.math.sqrt(self.inner_kernel.step_size)*(q1 - q0 - g)),
        list(current_transformed_state) if mcmc_util.is_list_like(current_transformed_state) else [
          current_transformed_state],
        list(proposed_transformed_state) if mcmc_util.is_list_like(proposed_transformed_state) else [
          proposed_transformed_state],
        previous_kernel_results.inner_results.accepted_results.diffusion_drift
      )
      log_accept_correction = kernel_results.inner_results.proposed_results.log_acceptance_correction
      kinetic_energy_fn = lambda q: -tfp.distributions.MultivariateNormalDiag(tf.zeros_like(q), tf.ones_like(q)).log_prob(q)
      kinetic_energy_difference = mcmc_util.safe_sum(tf.nest.map_structure(
        lambda v,g0,g1: kinetic_energy_fn(v+.5*(g0+g1)) - kinetic_energy_fn(v),
        eps,
        rep_transformed_current_log_target_grads,
        rep_transformed_proposed_log_target_grads
      ))
      total_energy_error = kinetic_energy_difference - proposed_transformed_target_log_prob + current_transformed_target_log_prob

      biased_speed_measure_loss = -tf.minimum(-total_energy_error, 0.) + self.proposal_entropy_weight * proposal_log_prob

      #tf.print('total_energy_error', total_energy_error)
      #tf.print('log_accept_ratio', log_accept_ratio)
      tf.print('grads',rep_transformed_current_log_target_grads)

      #speed_measure_loss = - tf.minimum(log_accept_ratio, 0.) + self.proposal_entropy_weight * proposal_log_prob
    grads_transformation = t1.gradient(speed_measure_loss, self.bijector_params)
    biased_grads_transformation = t1.gradient(biased_speed_measure_loss, self.bijector_params)

    #tf.print('grads_transformation', grads_transformation)
    #tf.print('biased_grads_transformation', biased_grads_transformation)

    safe_log_accept_ratio = tf.where(
      tf.math.is_finite(kernel_results.inner_results.log_accept_ratio),
      kernel_results.inner_results.log_accept_ratio,
      tf.constant(-np.inf, dtype = kernel_results.inner_results.log_accept_ratio.dtype))
    acceptance_rate = tf.reduce_mean(tf.math.exp(tf.minimum(0.,safe_log_accept_ratio)))
    #tf.print('acceptance_rate', acceptance_rate)
    #tf.print('proposal_entropy_weight' , self.proposal_entropy_weight)
    #tf.print('proposal_log_prob', proposal_log_prob)
    #adjust entropy weight based on acceptance rate
    beta = self.proposal_entropy_weight * (
        1. + self.learning_rate_beta * tf.cast(acceptance_rate - self.opt_acceptance_rate, tf.float32))
    beta = tf.clip_by_value(beta, self.min_entropy_weight, self.max_entropy_weight)
    self.proposal_entropy_weight.assign(beta)
    #grads_entropy_weight = - (acceptance_rate-self.opt_acceptance_rate)
    #self.optimizer_entropy_weight.apply_gradients(zip([grads_entropy_weight], [self.proposal_entropy_weight]))
    #projected gradient on positive value
    #self.proposal_entropy_weight.assign(tf.maximum(self.proposal_entropy_weight, 1e-5))

    #self.optimizer_transformation.apply_gradients(zip(grads_transformation, self.bijector_params))

    return next_state, kernel_results._replace(inner_results=kernel_results.inner_results._replace(
      extra = tf.nest.map_structure(lambda g1,g2: g1+g2,
                                    previous_kernel_results.inner_results.extra, grads_transformation)))

  def bootstrap_results(self, transformed_init_state):
    #we use the transformed init state only
    kernel_results = super().bootstrap_results(transformed_init_state = transformed_init_state)
    return kernel_results._replace(inner_results=kernel_results.inner_results._replace(
      extra = tf.nest.map_structure(lambda g: tf.zeros_like(g), self.bijector_params)))