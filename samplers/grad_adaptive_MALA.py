from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc import kernel as kernel_base
import numpy as np
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc import langevin



from tensorflow_probability.python.internal import prefer_static
from tensorflow.python.util import deprecation



class AdaptiveMALA(langevin.MetropolisAdjustedLangevinAlgorithm):
  def __init__(self, target_log_prob_fn, pre_cond_fn, pre_cond_params, x,
               learning_rate_beta, proposal_entropy_weight, optimizer,
               opt_acceptance_rate=.55, min_entropy_weight=.001, max_entropy_weight=100.,
               step_size=1., name=None, biased_grads = True):

    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UnadjustedAdaptiveMALA(
          target_log_prob_fn =  target_log_prob_fn,
          pre_cond_fn = pre_cond_fn,
          pre_cond_params = pre_cond_params,
          x = x,
          learning_rate_beta = learning_rate_beta,
          proposal_entropy_weight = proposal_entropy_weight,
          opt_acceptance_rate = opt_acceptance_rate,
          min_entropy_weight = min_entropy_weight,
          max_entropy_weight = max_entropy_weight,
          step_size = step_size,
          name = name,
          optimizer = optimizer,
          biased_grads = biased_grads))

    parameters = self._impl.inner_kernel.parameters.copy()
    self._parameters = parameters

  @property
  def is_calibrated(self):
    return True


  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  def one_step(self, current_state, previous_kernel_results):
    return self._impl.one_step(current_state, previous_kernel_results)

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    results = self._impl.bootstrap_results(init_state)
    return results


class UnadjustedAdaptiveMALA(langevin.UncalibratedLangevin):

  def __init__(self, target_log_prob_fn, pre_cond_fn, pre_cond_params, x,
               learning_rate_beta, proposal_entropy_weight, optimizer,
               opt_acceptance_rate=.55, min_entropy_weight=.001, max_entropy_weight=100., step_size=1.,
               name=None, biased_grads = True):

    self.pre_cond_fn = pre_cond_fn
    self.x = x
    self.proposal_entropy_weight = proposal_entropy_weight
    self.opt_acceptance_rate = opt_acceptance_rate
    self.learning_rate_beta = learning_rate_beta
    self.min_entropy_weight = min_entropy_weight
    self.max_entropy_weight = max_entropy_weight
    self.pre_cond_params = pre_cond_params
    self.biased_grads = biased_grads
    self.optimizer = optimizer

    self._parameters = dict(
      target_log_prob_fn = target_log_prob_fn,
      step_size = step_size,
      name = name,
      seed = None
    )

    super(UnadjustedAdaptiveMALA, self).__init__(
        target_log_prob_fn,
        step_size
    )


  @property
  def is_calibrated(self):
    return False


  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters


  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    return self._parameters['step_size']


  def one_step(self, current_state, previous_kernel_results):
    mala_step = super().one_step(current_state, previous_kernel_results)
    with tf.GradientTape(persistent = True) as t1:
      t1.watch(self.pre_cond_params)
      x=self.x
      t1.watch(x)
      pre_cond = self.pre_cond_fn(x)

      #draw noise distribution
      #if seed is not None:
      #  seed = samplers.sanitize_seed(seed)
      #else:
      #  seed = samplers.sanitize_seed(SeedStream(seed, salt='adaptiveMALA'))
      #seeds = samplers.split_seed(
      #  seed, n = 1, salt = 'adaptiveMALA.one_step')
      eps = samplers.normal(
          shape = tf.shape(current_state),
          dtype = dtype_util.base_dtype(current_state.dtype))

      #    seed = seed)

      #propose with a MALA step
      current_target_log_prob, current_grads_target_log_prob = tfp.math.value_and_gradient(
        self.target_log_prob_fn, current_state)
      next_state = current_state +.5*self.step_size**2* pre_cond.matvec(pre_cond.matvec(
        current_grads_target_log_prob, adjoint = True)) + self.step_size * pre_cond.matvec(eps)

      #grad of log target at next state
      reparam_next_target_log_prob, next_grads_target_log_prob = tfp.math.value_and_gradient(
        self.target_log_prob_fn, next_state)
      #compute momentum for accept ratio
      if self.biased_grads:
        #stop grad through target grads
        next_momentum = pre_cond.solvevec(eps, adjoint = True) + self.step_size / 2 * (
          current_grads_target_log_prob + tf.stop_gradient(next_grads_target_log_prob))
      else:
        next_momentum = pre_cond.solvevec(eps, adjoint = True) + self.step_size / 2 * (
          current_grads_target_log_prob + next_grads_target_log_prob)


      #energy error terms
      kinetic_energy_fn = lambda p: .5 * tf.einsum('...i,...i->...', pre_cond.matvec(p, adjoint=True),
                                                   pre_cond.matvec(p, adjoint=True))

      kinetic_energy_error = kinetic_energy_fn(next_momentum) - .5 * tf.einsum('...i,...i->...', eps, eps)
      log_acceptance_correction = - kinetic_energy_error

      target_log_prob, _ = mcmc_util.maybe_call_fn_and_grads(
        self.target_log_prob_fn, current_state, None, None)



      potential_energy_error = -(reparam_next_target_log_prob - current_target_log_prob)
      energy_error = potential_energy_error + kinetic_energy_error
      log_accept_rate = tf.minimum(tf.zeros([], potential_energy_error.dtype),
                                   -energy_error)

      d = tf.cast(tf.shape(current_state)[1:], x.dtype)

      proposal_log_prob = - d * tf.math.log(self.step_size) - pre_cond.log_abs_determinant()

      speed_measure_loss = - log_accept_rate + self.proposal_entropy_weight * proposal_log_prob

      #tf.print('speed_measure_loss',speed_measure_loss)
      #tf.print('proposal_log_prob',proposal_log_prob)
      #tf.print('log_accept_rate',log_accept_rate)
      #tf.print('log det', pre_cond.log_abs_determinant())
      #tf.print('C', pre_cond.to_dense())

    #compute grads for adaptation
    grads = t1.gradient(speed_measure_loss, self.pre_cond_params)
    grads = tf.nest.map_structure(
      lambda g: tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)), grads)
    #tf.print('grads', grads)

    self.grads = grads
    self.speed_measure_loss = speed_measure_loss

    # apply MCMC adaptation gradients
    self.optimizer.apply_gradients(zip(grads, self.pre_cond_params))

    #adjust entropy weight based on acceptance rate
    beta = self.proposal_entropy_weight * (
        1. + self.learning_rate_beta * tf.cast(tf.reduce_mean(tf.math.exp(log_accept_rate)) \
                                               - self.opt_acceptance_rate, tf.float32))
    beta = tf.where(tf.math.is_finite(beta), beta, self.proposal_entropy_weight)
    beta = tf.clip_by_value(beta, self.min_entropy_weight, self.max_entropy_weight)
    #tf.print('beta', beta)
    #tf.print('acceptance_rate', tf.reduce_mean(tf.math.exp(log_accept_rate)))
    self.proposal_entropy_weight.assign(beta)
    self.acceptance_rate = tf.reduce_mean(tf.math.exp(log_accept_rate))
    self.beta = beta

    kernel_results = previous_kernel_results._replace(
      log_acceptance_correction = log_acceptance_correction,
      target_log_prob = reparam_next_target_log_prob)

    return next_state , kernel_results



  def bootstrap_results(self, init_state):
    results = super().bootstrap_results(init_state)
    return results._replace(seed=[])