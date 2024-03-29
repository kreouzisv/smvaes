"""
Implementation of Hamiltonian Monte Carlo.
Currently only makes leapfrog moves with one step as that is all that is needed for HAIS.
"""

from packaging import version
import tensorflow as tf

#
# Configure TensorFlow depending on version
if version.parse(tf.__version__) >= version.parse('2.0.0'):
    # TensorFlow version 2
    tf = tf.compat.v1
elif version.parse(tf.__version__) >= version.parse('1.15'):
    tf.compat.v1.disable_eager_execution()
    tf = tf.compat.v1


def tf_expand_rank(input_, rank):
  "Expand the `input_` tensor to the given rank by appending dimensions"
  while len(input_.shape) < rank:
    input_ = tf.expand_dims(input_, axis=-1)
  return input_


def tf_expand_tile(input_, to_match):
  "Expand and tile the `input_` tensor to match the `to_match` tensor."
  assert len(input_.shape) <= len(to_match.shape)
  input_rank = len(input_.shape)
  match_rank = len(to_match.shape)
  tiling = [1] * input_rank + to_match.shape.as_list()[input_rank:]
  return tf.tile(tf_expand_rank(input_, match_rank), tiling)


def kinetic_energy(v, event_axes):
  """
  Calculate the kinetic energy of the system.
  :math:`- \\log \\Phi(v)` in Sohl-Dickstein and Culpepper's paper.
  Not normalised by :math:`M \\log(2 \\pi) / 2`
  """
  return 0.5 * tf.reduce_sum(tf.square(v), axis=event_axes)


def hamiltonian(position, velocity, energy_fn, event_axes):
  """
  Calculate the Hamiltonian of the system.
  Eqn 20 and 21 in Sohl-Dickstein and Culpepper's paper.
  """
  potential = energy_fn(position)
  momentum = kinetic_energy(velocity, event_axes)
  return potential + momentum


def mh_accept_reject(x0, v0, x1, v1, energy_fn, event_axes):
  """Accept or reject the leapfrog move according to Metropolis-Hastings.
  Step 3 in Sohl-Dickstein and Culpepper (2011).
  """
  E0 = hamiltonian(x0, v0, energy_fn, event_axes)
  E1 = hamiltonian(x1, -v1, energy_fn, event_axes)
  accept = metropolis_hastings_accept(E0=E0, E1=E1)
  # print('accept: {}'.format(accept.shape))
  # print('x0: {}'.format(x0.shape))
  # print('x1: {}'.format(x1.shape))
  # Expand the accept (which has batch shape) to full (batch + event) shape.
  accept_tiled = tf_expand_tile(accept, x1)
  xdash = tf.where(accept_tiled, x1, x0)
  vdash = tf.where(accept_tiled, -v1, v0)
  # print('xdash: {}'.format(xdash.shape))
  return xdash, vdash, accept


def metropolis_hastings_accept(E0, E1):
  """
  Accept or reject a move based on the energies of the two states.
  """
  ediff = E0 - E1
  return ediff >= tf.math.log(tf.random.uniform(shape=tf.shape(ediff)))


def leapfrog(x0, v0, eps, energy_fn):
  """
  Simulate the Hamiltonian dynamics using leapfrog method.
  That is follow the 2nd step in the 5 step
  procedure in Section 2.3 of Sohl-Dickstein and Culpepper's paper.
  Note this leapfrog procedure only has one step.
  """
  eps = tf.convert_to_tensor(eps)
  epshalf = tf_expand_tile(eps / 2., v0)
  xhalf = x0 + epshalf * v0
  dE_dx = tf.gradients(tf.reduce_sum(energy_fn(xhalf)), xhalf)[0]
  v1 = v0 - tf_expand_tile(eps, v0) * dE_dx
  x1 = xhalf + epshalf * v1
  return x1, v1


def default_gamma(eps):
  """Calculate the default gamma (momentum refresh parameter).
  Follows equation 11. in Culpepper et al. (2011)
  """
  return 1. - tf.math.exp(eps * tf.math.log(1 / 2.))


def hmc_move(x0, v0, energy_fn, event_axes, eps, gamma=None):
  """
  Make a HMC move.
  Implements the algorithm in
  Culpepper et al. 2011 "Building a better probabilistic model of images by factorization".
  Args:
    gamma: Set to 1 to remove any partial momentum refresh (momentum is sampled fresh every move)
  """
  #
  # STEP 2:
  # Simulate the dynamics of the system using leapfrog
  x1, v1 = leapfrog(x0=x0, v0=v0, eps=eps, energy_fn=energy_fn)
  #
  # STEP 3:
  # Accept or reject according to MH
  xdash, vdash, accept = mh_accept_reject(x0, v0, x1, v1, energy_fn, event_axes)
  #
  # STEP 4:
  # Partial momentum refresh.
  # gamma is the parameter governing this
  if gamma is None:
    gamma = default_gamma(eps)
  vtilde = partial_momentum_refresh(vdash, gamma)
  #
  # Return state
  return accept, xdash, vtilde


def partial_momentum_refresh(vdash, gamma):
  """Update vdash with a partial momentum refresh.
  Step 4 in Sohl-Dickstein and Culpepper (2011).
  """
  # There is some disagreement between the above paper and the description of STEP 4.
  # Specifically the second sqrt below is omitted in the description of STEP 4.
  r = tf.random.normal(tf.shape(vdash))
  gamma = tf_expand_tile(gamma, vdash)
  return - tf.sqrt(1 - gamma) * vdash + tf.sqrt(gamma) * r


def smooth_acceptance_rate(accept, old_acceptance_rate, acceptance_decay):
  #
  # Smooth the acceptance rate
  assert accept.shape == old_acceptance_rate.shape
  new_acceptance_rate = tf.add(
      acceptance_decay * old_acceptance_rate,
      (1.0 - acceptance_decay) * tf.cast(accept, old_acceptance_rate.dtype, name='cast_accept'))
  return new_acceptance_rate


def hmc_sample(x0, log_target, eps, sample_shape=(), event_axes=(), v0=None,
               niter=1000, nchains=3000, acceptance_decay=.9):
  """Sample using Hamiltonian Monte Carlo.
  Args:
    x0: Initial state
    log_target: The unnormalised target log density
    eps: Step size for HMC
    sample_shape: The shape of the samples, e.g. `()` for univariate or (3,) a 3-dimensional MVN
    event_axes: Index into `x0`'s dimensions for individual samples, `()` for univariate sampling
    v0: Initial velocity, will be sampled if None
    niter: Number of iterations in each chain
    nchains: Number of chains to run in parallel
    acceptance_decay: Decay used to calculate smoothed acceptance rate
  Returns:
    A tuple (final state, final velocity, the samples, the smoothed acceptance rate)
  """
  def condition(i, x, v, samples, smoothed_accept_rate):
    "The condition keeps the while loop going until we have finished the iterations."
    return tf.less(i, niter)

  def body(i, x, v, samples, smoothed_accept_rate):
    "The body of the while loop over the iterations."
    #
    # New step: make a HMC move
    accept, xnew, vnew = hmc_move(
        x,
        v,
        energy_fn=lambda x: -log_target(x),
        event_axes=event_axes,
        eps=eps,
    )
    #
    # Update the TensorArray storing the samples
    samples = samples.write(i, xnew)
    #
    # Smooth the acceptance rate
    smoothed_accept_rate = smooth_acceptance_rate(accept, smoothed_accept_rate, acceptance_decay)
    #
    return tf.add(i, 1), xnew, vnew, samples, smoothed_accept_rate

  #
  # Sample initial velocity if not provided
  if v0 is None:
    v0 = tf.random_normal(tf.shape(x0))
  #
  # Keep the samples in a TensorArray
  samples = tf.TensorArray(dtype=x0.dtype, size=niter, element_shape=(nchains,) + sample_shape)
  #
  # Current iteration
  iteration = tf.constant(0)
  #
  # Smoothed acceptance rate
  smoothed_accept_rate = tf.constant(.65, shape=(nchains,), dtype=tf.float32)
  #
  # Current step size and adjustments
  # stepsize = tf.constant(STEPSIZE_INITIAL, shape=(NCHAINS,), dtype=tf.float32)
  # stepsize_dec = STEPSIZE_DEC * tf.ones(smoothed_acceptance_rate.shape)
  # stepsize_inc = STEPSIZE_INC * tf.ones(smoothed_acceptance_rate.shape)
  #
  # While loop across iterations
  n, x, v, samples_final, smoothed_accept_rate_final = \
      tf.while_loop(
          condition,
          body,
          (iteration, x0, v0, samples, smoothed_accept_rate),
          parallel_iterations=1,
          swap_memory=True)
  #
  return x, v, samples_final, smoothed_accept_rate_final