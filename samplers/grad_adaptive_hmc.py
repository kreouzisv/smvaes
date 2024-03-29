from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc import hmc

import samplers.leapfrog_integrator_trajectory as leapfrog_impl
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.random import rademacher
from tensorflow_probability.python.distributions import Geometric, Normal
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds



class AdaptiveHMC(hmc.HamiltonianMonteCarlo):
  def __init__(self, target_log_prob_fn,
               num_leapfrog_steps,
               pre_cond_fn, pre_cond_params, x,
               learning_rate_beta, proposal_entropy_weight, penalty_param, optimizer,
               opt_acceptance_rate=.65, min_entropy_weight=.001, max_entropy_weight=100.,
               step_size=1., name=None, biased_grads = True):

    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UnadjustedAdaptiveHMC(
          target_log_prob_fn = target_log_prob_fn,
          num_leapfrog_steps = num_leapfrog_steps,
          pre_cond_fn = pre_cond_fn,
          pre_cond_params = pre_cond_params,
          x = x,
          learning_rate_beta = learning_rate_beta,
          penalty_param = penalty_param,
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


class UnadjustedAdaptiveHMC(hmc.UncalibratedHamiltonianMonteCarlo):

  def __init__(self, num_leapfrog_steps,
               target_log_prob_fn, pre_cond_fn, pre_cond_params, x,
               learning_rate_beta, proposal_entropy_weight, penalty_param, optimizer,
               opt_acceptance_rate=.65, min_entropy_weight=.001, max_entropy_weight=100., step_size=1.,
               name=None, biased_grads = True):

    self.pre_cond_fn = pre_cond_fn
    self.x = x
    self.beta = proposal_entropy_weight
    self.opt_acceptance_rate = opt_acceptance_rate
    self.learning_rate_beta = learning_rate_beta
    self.min_entropy_weight = min_entropy_weight
    self.max_entropy_weight = max_entropy_weight
    self.pre_cond_params = pre_cond_params
    self.biased_accept_grads = biased_grads
    self.optimizer = optimizer


    self._lipschitz_threshold_jvp = .99
    self._num_exact_trace_terms = 2
    self._num_trace_terms_probs = .5


    self._penalty_param = penalty_param
    self._clip_grad_value = 1e4
    self._max_penalty = 1e5

    def penalty_fn(evs):
        delta_1 = .75
        delta_2 = 1.75
        return tf.nest.map_structure(
          lambda ev: tf.where(abs(ev) < delta_2,
                              tf.where(abs(ev) < delta_1,
                                       tf.zeros_like(ev),
                                       (abs(ev) - delta_1) ** 2),
                              (delta_2 - delta_1) ** 2 + (delta_2 - delta_1) * (abs(ev) - delta_2)
                              ),
          evs
        )
    self._penalty_fn = penalty_fn


    self._parameters = dict(
      target_log_prob_fn = target_log_prob_fn,
      step_size = step_size,
      name = name,
      seed = None
    )

    super(UnadjustedAdaptiveHMC, self).__init__(
        target_log_prob_fn,
        step_size,
        num_leapfrog_steps = num_leapfrog_steps
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


  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'hmc', 'one_step')):

      with tf.GradientTape(persistent = False) as tape1:

        step_size = self.step_size
        num_leapfrog_steps = self.num_leapfrog_steps

        tape1.watch(self.pre_cond_params)
        x=self.x
        tape1.watch(x)
        pre_cond_operator = self.pre_cond_fn(x)
        momentum_distribution = tfp.experimental.distributions.MultivariateNormalPrecisionFactorLinearOperator(
          loc = tf.zeros_like(current_state),
          precision_factor = pre_cond_operator
        )

        [
            current_state_parts,
            step_sizes,
            momentum_distribution,
            pre_cond_operator,
            current_target_log_prob,
            current_target_log_prob_grad_parts,
        ] = _prepare_args(
            self.target_log_prob_fn,
            current_state,
            step_size,
            momentum_distribution,
            pre_cond_operator,
            previous_kernel_results.target_log_prob,
            previous_kernel_results.grads_target_log_prob,
            maybe_expand=True,
            state_gradients_are_stopped=self.state_gradients_are_stopped)

        seed = samplers.sanitize_seed(seed)
        seeds = samplers.split_seed(seed, n = len(current_state_parts))

        current_momentum_noise_parts = tf.nest.map_structure(
         lambda x,s: tf.cast(samplers.normal(shape=tf.shape(x), seed=s), x.dtype),
          current_state_parts, seeds)
        tape1.watch(current_momentum_noise_parts)

        current_momentum_parts = tf.nest.map_structure(
         lambda C,v: C.solvevec(v, adjoint = True),
          pre_cond_operator, current_momentum_noise_parts)

        momentum_log_prob = getattr(momentum_distribution,
                                  '_log_prob_unnormalized',
                                  momentum_distribution.log_prob)
        kinetic_energy_fn = lambda *args: -momentum_log_prob(*args)

        leapfrog_kinetic_energy_fn = kinetic_energy_fn

        integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
            self.target_log_prob_fn, step_sizes, num_leapfrog_steps)

        [
            next_momentum_parts,
            next_state_parts,
            next_target_log_prob,
            next_target_log_prob_grad_parts,
            momentum_parts_array,
            target_log_prob_grad_parts_array_non_stopped,
            state_parts_array
        ] = integrator(
            current_momentum_parts,
            current_state_parts,
            target=current_target_log_prob,
            target_grad_parts=current_target_log_prob_grad_parts,
            kinetic_energy_fn=leapfrog_kinetic_energy_fn,
            )
        if self.state_gradients_are_stopped:
          next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]

        def maybe_flatten(x):
          return x if mcmc_util.is_list_like(current_state) else x[0]

        #linear/langevin part for proposal log prob
        dims = tf.nest.map_structure(
          lambda x: tf.cast(tf.shape(x)[1:], x.dtype), current_state_parts)
        log_det_linear_term = tf.nest.map_structure(
          lambda d, h, C: d * tf.math.log(h*self.num_leapfrog_steps
                                          ) + C.log_abs_determinant(),
          dims, step_sizes, pre_cond_operator)

        if not self.biased_accept_grads:
          log_acceptance_correction = _compute_log_acceptance_correction(
                  kinetic_energy_fn, current_momentum_parts,
                  next_momentum_parts)
          #compute log-acceptance rates
          to_sum = [next_target_log_prob,
                    -previous_kernel_results.target_log_prob,
                    log_acceptance_correction]
          log_accept_ratio = mcmc_util.safe_sum(
            to_sum, name = 'log_accept_ratio')
          log_accept_ratio_loss = -tf.minimum(tf.zeros([], dtype = log_accept_ratio.dtype), log_accept_ratio)

        ###
        #compute biased objective for optimizing log accept ratio
        ###
        # stop gradients
        target_log_prob_grad_parts_array = tf.nest.map_structure(lambda x: tf.stop_gradient(x),
                                                                 target_log_prob_grad_parts_array_non_stopped)
        sum_potential_grads = tf.nest.map_structure(
          lambda g: -.5 * (g[0] + g[-1]) - tf.reduce_sum(g[1:-1], 0), target_log_prob_grad_parts_array)
        kinetic_energy_error_stopped = mcmc_util.safe_sum(tf.nest.map_structure(
          lambda C, h, Us, v: (kinetic_energy_fn(C.solvevec(v, adjoint = True) - h * tf.stop_gradient(Us)
                                                 ) - kinetic_energy_fn(C.solvevec(v, adjoint = True))),
          pre_cond_operator, step_sizes, sum_potential_grads, current_momentum_noise_parts))
        log_acceptance_correction_stopped = tf.nest.map_structure(lambda x: -x, kinetic_energy_error_stopped)


        weights = tf.cast(num_leapfrog_steps - tf.range(1, num_leapfrog_steps),
                          current_momentum_noise_parts[0].dtype)
        xi = tf.nest.map_structure(
          lambda g: tf.einsum('lsd,l->sd', -g[1:-1], weights), target_log_prob_grad_parts_array_non_stopped)
        reparam_next_state_xi = tf.nest.map_structure(
          lambda x, v, C, h, g, g0: x+h*num_leapfrog_steps*C.matvec(v) -h*h*C.matvec(
            C.matvec(tf.stop_gradient(g), adjoint = True)) + h*h/2*num_leapfrog_steps*C.matvec(
            C.matvec(tf.stop_gradient(g0), adjoint = True)),
          current_state_parts, current_momentum_noise_parts, pre_cond_operator, step_sizes,
          xi, current_target_log_prob_grad_parts)
        reparam_next_target_log_prob_xi, _ = mcmc_util.maybe_call_fn_and_grads(
          self.target_log_prob_fn, reparam_next_state_xi, None, None)
        potential_energy_error_stopped_xi = -(reparam_next_target_log_prob_xi - current_target_log_prob)
        log_accept_loss_stopped = - tf.minimum(tf.zeros([], potential_energy_error_stopped_xi.dtype),
                                           -potential_energy_error_stopped_xi - kinetic_energy_error_stopped)
        log_accept_ratio_stopped = -potential_energy_error_stopped_xi - kinetic_energy_error_stopped


        #############
        #Approximate DS matrix using a constant Hessian at the mid-point
        #############

        def approximate_DS_matrix_vector_first_order(ws):
          approx_DSw = tf.nest.map_structure(
            lambda h, C, qs, w: h * h * tf.cast(self.num_leapfrog_steps ** 2 - 1, qs.dtype) / 6. * C.matvec(
              _target_hessian_vector_product(
                self.target_log_prob_fn,
                qs[self.num_leapfrog_steps//2],
                C.matvec(w)),adjoint = True),
            step_sizes, pre_cond_operator, state_parts_array, ws)
          return approx_DSw


        if self.num_leapfrog_steps>1:
          DSw_approximation = approximate_DS_matrix_vector_first_order
          ###########
          # Russian Roulette estimator for the residual part
          ###########
          # distribution of truncation level
          random_trace_terms_dist = Geometric(probs = self._num_trace_terms_probs)
          coeff_fn = lambda k: 1. / (1 - random_trace_terms_dist.cdf(
            k - self._num_exact_trace_terms + .1))
          trace_noise_parts = tf.nest.map_structure(
            lambda x, s: rademacher(shape = tf.shape(x), seed = s, dtype = x.dtype), current_state_parts, seeds)

          def loop_residual_trace_neumann(k, prev_jvp, ns_jvp, sum_jvp_entropy):
            new_jvp = DSw_approximation(prev_jvp)
            # clip values
            prev_jvp_norm = tf.nest.map_structure(lambda x: tf.linalg.norm(x, axis = -1, keepdims = True), prev_jvp)
            new_jvp_norm = tf.nest.map_structure(lambda x: tf.linalg.norm(x, axis = -1, keepdims = True), new_jvp)
            new_jvp = tf.nest.map_structure(
             lambda x, y, z: x * tf.minimum(tf.ones([], dtype =y.dtype), self._lipschitz_threshold_jvp * y / z),
             new_jvp, prev_jvp_norm, new_jvp_norm)

            new_ns_jvp = tf.nest.map_structure(
              lambda ns, jvp: ns + tf.cast(tf.pow(-1., k), ns.dtype) * tf.cast(coeff_fn(k), ns.dtype) * jvp, ns_jvp,
              new_jvp)
            new_sum_vjp_entropy = tf.nest.map_structure(
              lambda ent, jvp: ent + tf.cast(tf.pow(-1., k + 1.) / k * coeff_fn(k), jvp.dtype) * jvp,
              sum_jvp_entropy, new_jvp)

            return k + 1, new_jvp, new_ns_jvp, new_sum_vjp_entropy

          # random truncation level for russian roulette estimator
          sample_random_trace_terms = random_trace_terms_dist.sample(seed = seeds[0])
          loop_trace_noise_parts = trace_noise_parts
          with tape1.stop_recording():
            with tf.name_scope('russian_roulette_estimator'):
              _, jvp_residual, ns_jvp_residual, sum_jvp_residual_entropy = tf.while_loop(
                cond = lambda k, _0, _1, _2: k <= self._num_exact_trace_terms + sample_random_trace_terms,
                body = loop_residual_trace_neumann,
                loop_vars = [tf.constant(1., dtype = tf.float32),
                             loop_trace_noise_parts,
                             loop_trace_noise_parts,
                             [tf.zeros_like(ep) for ep in loop_trace_noise_parts]
                             ]
              )


          # power itertation for penalty lipschitz constant estimate
          # add some noise for stability, the noise might be dominant for very small jvps,
          # but then we should have a contraction, also clip value
          stable_jvp_residual = tf.nest.map_structure(lambda w: tf.clip_by_value(w + tf.cast(1e-6 * samplers.normal(
            w.shape), w.dtype), -1e6, 1e6), jvp_residual)
          normalised_power_iteration_vector = tf.nest.map_structure(
            lambda w: tf.stop_gradient(w / tf.linalg.norm(w, axis = -1, keepdims = True)), stable_jvp_residual)
          jvp_lipschitz = DSw_approximation(normalised_power_iteration_vector)
          #lipschitz_constant_estimate = tf.nest.map_structure(
          #  lambda j: tf.norm(j, axis = -1), jvp_lipschitz)

          max_eigenvalue_DS_lipschitz = tf.nest.map_structure(
            lambda w, DSw: tf.einsum('si,si->s', w, DSw),
            normalised_power_iteration_vector, jvp_lipschitz)

          # JVP using AD through tape
          jvp_noise =DSw_approximation(trace_noise_parts)
          neumann_trace_residual_for_grad = tf.nest.map_structure(
            lambda ns, j: tf.einsum('ij,ij->i', tf.stop_gradient(ns), j),
            ns_jvp_residual, jvp_noise)

          penalty = tf.nest.map_structure(lambda x: tf.cast(self._penalty_param, x.dtype)*x,
                                          self._penalty_fn(max_eigenvalue_DS_lipschitz))
          trace_residual_log_det = tf.nest.map_structure(
            lambda s, eps: tf.einsum('ij,ij->i', tf.stop_gradient(s), eps), sum_jvp_residual_entropy,
            trace_noise_parts)

          trace_estimate_for_grad = tf.nest.map_structure(lambda x,y: x-y,
                                                          neumann_trace_residual_for_grad, penalty)

        else:
          trace_estimate_for_grad = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0],c.dtype), next_state_parts)
          trace_residual_log_det = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0],c.dtype), next_state_parts)
          penalty = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0],c.dtype), next_state_parts)
          max_eigenvalue_DS_lipschitz = tf.nest.map_structure(lambda c: tf.zeros(c.shape[0],c.dtype), next_state_parts)

        ######
        # proposal entropy terms and loss functions
        ######
        noise_proposal_log_probs = tf.nest.map_structure(
          lambda v: tf.reduce_sum(Normal(loc = 0.*tf.ones([], v.dtype), scale = 1.).log_prob(v), -1),
          current_momentum_noise_parts)

        proposal_log_prob_estimate = mcmc_util.safe_sum(tf.nest.map_structure(
          lambda y, z, c: - y - z + c,
          log_det_linear_term, trace_residual_log_det, noise_proposal_log_probs))

        proposal_log_prob_for_grad = mcmc_util.safe_sum(tf.nest.map_structure(
          lambda y, z: - y - z,
          log_det_linear_term, trace_estimate_for_grad))

        #loss for optimizing
        if self.biased_accept_grads:
          mcmc_loss = log_accept_loss_stopped + \
                    tf.cast(self.beta * tf.ones([]), log_accept_loss_stopped.dtype) * proposal_log_prob_for_grad
        else:
          mcmc_loss = log_accept_ratio_loss + \
          tf.cast(self.beta * tf.ones([]), log_accept_ratio_loss.dtype) * proposal_log_prob_for_grad



      grads = tape1.gradient(mcmc_loss, self.pre_cond_params)



      #replace nans
      grads = tf.nest.map_structure(
        lambda g: tf.where(tf.math.is_nan(g), tf.zeros_like(g), g), grads)
      #clip grads
      grads = tf.nest.map_structure(lambda g: tf.clip_by_value(g, -self._clip_grad_value, self._clip_grad_value),
                                    grads)

      

    # apply MCMC adaptation gradients
    self.optimizer.apply_gradients(zip(grads, self.pre_cond_params))

    #adjust entropy weight based on acceptance rate
    if self.biased_accept_grads:
      acceptance_rate = tf.reduce_mean(tf.math.exp(tf.minimum(tf.zeros([], dtype = log_accept_ratio_stopped.dtype),
                                                              log_accept_ratio_stopped)))
    else:
      acceptance_rate = tf.reduce_mean(tf.math.exp(tf.minimum(tf.zeros([], dtype = log_accept_ratio.dtype),
                                                              log_accept_ratio)))

    beta = self.beta * (
        1. + self.learning_rate_beta * tf.cast(tf.reduce_mean(acceptance_rate) \
                                               - self.opt_acceptance_rate, tf.float32))
    beta = tf.where(tf.math.is_finite(beta), beta, self.beta)
    beta = tf.clip_by_value(beta, self.min_entropy_weight, self.max_entropy_weight)
    #tf.print('beta', beta)
    
    self.beta.assign(beta)

    # update penalty param
    current_penalty_param = self._penalty_param
    new_penalty_param = tf.clip_by_value(
      current_penalty_param + self.learning_rate_beta * tf.clip_by_value(
        tf.cast(tf.reduce_mean(penalty), tf.float32), 0, self._max_penalty), 1., self._max_penalty)
    new_penalty_param = tf.where(tf.math.is_nan(new_penalty_param),
                                 current_penalty_param,
                                 new_penalty_param)
    self._penalty_param.assign(new_penalty_param)

    new_kernel_results = previous_kernel_results._replace(
      log_acceptance_correction = log_acceptance_correction if not self.biased_accept_grads \
        else log_acceptance_correction_stopped,
      target_log_prob = next_target_log_prob,
      grads_target_log_prob = next_target_log_prob_grad_parts,
      initial_momentum = current_momentum_parts,
      final_momentum = next_momentum_parts,
      seed = seed,
            )

    self.grads = grads
    self.speed_measure_loss = mcmc_loss
    #self.beta = beta
    self.acceptance_rate = acceptance_rate



    #tf.print('acceptance_rate', acceptance_rate)
    #tf.print('grads', grads)
    # tf.print('params', self.pre_cond_params)

    return maybe_flatten(next_state_parts), new_kernel_results


  # def one_step(self, current_state, previous_kernel_results):
  #   #mala_step = super().one_step(current_state, previous_kernel_results)
  #   with tf.GradientTape(persistent = True) as t1:
  #     t1.watch(self.pre_cond_params)
  #     x=self.x
  #     t1.watch(x)
  #     pre_cond = self.pre_cond_fn(x)
  #
  #     #draw noise distribution
  #     #if seed is not None:
  #     #  seed = samplers.sanitize_seed(seed)
  #     #else:
  #     #  seed = samplers.sanitize_seed(SeedStream(seed, salt='adaptiveMALA'))
  #     #seeds = samplers.split_seed(
  #     #  seed, n = 1, salt = 'adaptiveMALA.one_step')
  #     eps = samplers.normal(
  #         shape = tf.shape(current_state),
  #         dtype = dtype_util.base_dtype(current_state.dtype))
  #
  #     #    seed = seed)
  #
  #     #propose with a MALA step
  #     current_target_log_prob, current_grads_target_log_prob = tfp.math.value_and_gradient(
  #       self.target_log_prob_fn, current_state)
  #     next_state = current_state +.5*self.step_size**2* pre_cond.matvec(pre_cond.matvec(
  #       current_grads_target_log_prob, adjoint = True)) + self.step_size * pre_cond.matvec(eps)
  #
  #     #grad of log target at next state
  #     reparam_next_target_log_prob, next_grads_target_log_prob = tfp.math.value_and_gradient(
  #       self.target_log_prob_fn, next_state)
  #     #compute momentum for accept ratio
  #     if self.biased_grads:
  #       #stop grad through target grads
  #       next_momentum = pre_cond.solvevec(eps, adjoint = True) + self.step_size / 2 * (
  #         current_grads_target_log_prob + tf.stop_gradient(next_grads_target_log_prob))
  #     else:
  #       next_momentum = pre_cond.solvevec(eps, adjoint = True) + self.step_size / 2 * (
  #         current_grads_target_log_prob + next_grads_target_log_prob)
  #
  #
  #     #energy error terms
  #     kinetic_energy_fn = lambda p: .5 * tf.einsum('...i,...i->...', pre_cond.matvec(p, adjoint=True),
  #                                                  pre_cond.matvec(p, adjoint=True))
  #     kinetic_energy_error = kinetic_energy_fn(next_momentum) - .5 * tf.einsum('...i,...i->...', eps, eps)
  #     log_acceptance_correction = - kinetic_energy_error
  #
  #     target_log_prob, _ = mcmc_util.maybe_call_fn_and_grads(
  #       self.target_log_prob_fn, current_state, None, None)
  #
  #     potential_energy_error = -(reparam_next_target_log_prob - current_target_log_prob)
  #     energy_error = potential_energy_error + kinetic_energy_error
  #     log_accept_rate = tf.minimum(tf.zeros([], potential_energy_error.dtype),
  #                                  -energy_error)
  #
  #     d = tf.cast(tf.shape(current_state)[1:], x.dtype)
  #
  #     proposal_log_prob = - d * tf.math.log(self.step_size) - pre_cond.log_abs_determinant()
  #
  #     speed_measure_loss = - log_accept_rate + self.proposal_entropy_weight * proposal_log_prob
  #
  #     #tf.print('speed_measure_loss',speed_measure_loss)
  #     tf.print('proposal_log_prob',proposal_log_prob)
  #     tf.print('log_accept_rate',log_accept_rate)
  #     tf.print('log det', pre_cond.log_abs_determinant())
  #     #tf.print('C', pre_cond.to_dense())
  #
  #   #compute grads for adaptation
  #   grads = t1.gradient(speed_measure_loss, self.pre_cond_params)
  #   grads = tf.nest.map_structure(
  #     lambda g: tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)), grads)
  #   #tf.print('grads', grads)
  #
  #   # apply MCMC adaptation gradients
  #   self.optimizer.apply_gradients(zip(grads, self.pre_cond_params))
  #
  #   #adjust entropy weight based on acceptance rate
  #   beta = self.proposal_entropy_weight * (
  #       1. + self.learning_rate_beta * tf.cast(tf.reduce_mean(tf.math.exp(log_accept_rate)) \
  #                                              - self.opt_acceptance_rate, tf.float32))
  #   beta = tf.where(tf.math.is_finite(beta), beta, self.proposal_entropy_weight)
  #   beta = tf.clip_by_value(beta, self.min_entropy_weight, self.max_entropy_weight)
  #   tf.print('beta', beta)
  #   self.proposal_entropy_weight.assign(beta)
  #
  #   kernel_results = previous_kernel_results._replace(
  #     log_acceptance_correction = log_acceptance_correction,
  #     target_log_prob = reparam_next_target_log_prob)
  #
  #   return next_state , kernel_results
  #


  #def bootstrap_results(self, init_state):
  #  results = super().bootstrap_results(init_state)
  #  return results._replace(seed=[])




def _compute_log_acceptance_correction(kinetic_energy_fn,
                                       current_momentums,
                                       proposed_momentums,
                                       name=None):
  """Helper to `kernel` which computes the log acceptance-correction.
  A sufficient but not necessary condition for the existence of a stationary
  distribution, `p(x)`, is "detailed balance", i.e.:
  ```none
  p(x'|x) p(x) = p(x|x') p(x')
  ```
  In the Metropolis-Hastings algorithm, a state is proposed according to
  `g(x'|x)` and accepted according to `a(x'|x)`, hence
  `p(x'|x) = g(x'|x) a(x'|x)`.
  Inserting this into the detailed balance equation implies:
  ```none
      g(x'|x) a(x'|x) p(x) = g(x|x') a(x|x') p(x')
  ==> a(x'|x) / a(x|x') = p(x') / p(x) [g(x|x') / g(x'|x)]    (*)
  ```
  One definition of `a(x'|x)` which satisfies (*) is:
  ```none
  a(x'|x) = min(1, p(x') / p(x) [g(x|x') / g(x'|x)])
  ```
  (To see that this satisfies (*), notice that under this definition only at
  most one `a(x'|x)` and `a(x|x') can be other than one.)
  We call the bracketed term the "acceptance correction".
  In the case of UncalibratedHMC, the log acceptance-correction is not the log
  proposal-ratio. UncalibratedHMC augments the state-space with momentum, z.
  Given a probability density of `m(z)` for momentums, the chain eventually
  converges to:
  ```none
  p([x, z]) propto= target_prob(x) m(z)
  ```
  Relating this back to Metropolis-Hastings parlance, for HMC we have:
  ```none
  p([x, z]) propto= target_prob(x) m(z)
  g([x, z] | [x', z']) = g([x', z'] | [x, z])
  ```
  In other words, the MH bracketed term is `1`. However, because we desire to
  use a general MH framework, we can place the momentum probability ratio inside
  the metropolis-correction factor thus getting an acceptance probability:
  ```none
                       target_prob(x')
  accept_prob(x'|x) = -----------------  [m(z') / m(z)]
                       target_prob(x)
  ```
  (Note: we actually need to handle the kinetic energy change at each leapfrog
  step, but this is the idea.)
  For consistency, we compute this correction in log space, using the kinetic
  energy function, `K(z)`, which is the negative log probability of the momentum
  distribution. So the log acceptance probability is
  ```none
  log(correction) = log(m(z')) - log(m(z))
                  = K(z) - K(z')
  ```
  Note that this is equality, since the normalization constants on `m` cancel
  out.
  Args:
    kinetic_energy_fn: Python callable that can evaluate the kinetic energy
      of the given momentum. This is typically the negative log probability of
      the distribution over the momentum.
    current_momentums: (List of) `Tensor`s representing the value(s) of the
      current momentum(s) of the state (parts).
    proposed_momentums: (List of) `Tensor`s representing the value(s) of the
      proposed momentum(s) of the state (parts).
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'compute_log_acceptance_correction').
  Returns:
    log_acceptance_correction: `Tensor` representing the `log`
      acceptance-correction.  (See docstring for mathematical definition.)
  """
  with tf.name_scope(name or 'compute_log_acceptance_correction'):
    current_kinetic = kinetic_energy_fn(current_momentums)
    proposed_kinetic = kinetic_energy_fn(proposed_momentums)
    return mcmc_util.safe_sum([current_kinetic, -proposed_kinetic])


def _prepare_args(target_log_prob_fn,
                  state,
                  step_size,
                  momentum_distribution,
                  pre_cond_operator,
                  target_log_prob=None,
                  grads_target_log_prob=None,
                  maybe_expand=False,
                  state_gradients_are_stopped=False):
  """Helper which processes input args to meet list-like assumptions."""
  state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
  if state_gradients_are_stopped:
    state_parts = [tf.stop_gradient(x) for x in state_parts]
  target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
      target_log_prob_fn, state_parts, target_log_prob, grads_target_log_prob)
  step_sizes, _ = mcmc_util.prepare_state_parts(
      step_size, dtype=target_log_prob.dtype, name='step_size')


  # The momentum will get "maybe listified" to zip with the state parts,
  # and this step makes sure that the momentum distribution will have the
  # same "maybe listified" underlying shape.
  if not mcmc_util.is_list_like(momentum_distribution.dtype):
    momentum_distribution = jds.JointDistributionSequential(
        [momentum_distribution])

  if not mcmc_util.is_list_like(pre_cond_operator.dtype):
    pre_cond_operator = [pre_cond_operator]

  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      momentum_distribution,
      pre_cond_operator,
      target_log_prob,
      grads_target_log_prob,
  ]

stop_hessian_grad = True
def _target_hessian_vector_product(target_log_prob_fn, q, w):
  if stop_hessian_grad:
    q = tf.stop_gradient(q)
  with tf.GradientTape() as tape1:
    tape1.watch(q)
    with tf.GradientTape() as tape2:
      tape2.watch(q)
      y = target_log_prob_fn(q)
    grads = tape2.gradient(y, q)
  hvp = tape1.gradient(grads, q, output_gradients = w)
  return hvp