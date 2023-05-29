import matplotlib.pyplot as plt
import os
import tensorflow as tf


def make_bijector_kwargs(bijector, name_to_kwargs):
  import re
  
  if hasattr(bijector, 'bijectors'):
    return {b.name: make_bijector_kwargs(b, name_to_kwargs) for b in bijector.bijectors}
  else:
    for name_regex, kwargs in name_to_kwargs.items():
      if re.match(name_regex, bijector.name):
        return kwargs
  return {}


from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

tfk = tf.keras
tfkl = tf.keras.layers


class PseudoInputs(tfkl.Layer):
  def __init__(self, n_pseudo_inputs=1000):
      super().__init__()
      self.n_pseudo_inputs = n_pseudo_inputs
      self.pseudo_inputs = None

  def call(self, inputs=None):
      # abstract method
      pass

  def get_n(self):
      return self.n_pseudo_inputs


class PInputsGenerated(PseudoInputs):
  def __init__(self, original_dim=(32, 32, 3), pseudo_inputs_mean=0.0,
               pseudo_inputs_std=0.01, n_pseudo_inputs=500):
      super().__init__(n_pseudo_inputs)
      self.pseudo_input_std = pseudo_inputs_std
      self.pseudo_input_mean = pseudo_inputs_mean
      self.pre_pseudo_inputs = tf.eye(n_pseudo_inputs)

      self.pseudo_inputs_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(np.prod(original_dim),kernel_initializer=tfk.initializers.RandomNormal(mean=pseudo_inputs_mean, stddev=pseudo_inputs_std),activation="relu"),
        tf.keras.layers.Reshape(original_dim)
        ])

  def call(self, inputs=None):
      # If the pre pseudo inputs are generated we have to create the new
      # pseudo inputs
      #
      # In case of the "generate" vampprior an additional fully
      # connected layer maps a scalar to an image
      # (e.g in case of mnist [] -> [28, 28, 1] )
      # recompute pseudo inputs if we choose the generate strategy
      return self.pseudo_inputs_layer(self.pre_pseudo_inputs)


class PInputsData(PseudoInputs):
  def __init__(self, pseudo_inputs):
      super().__init__(n_pseudo_inputs=pseudo_inputs.shape[0])
      self.pseudo_inputs = pseudo_inputs

  def call(self, inputs=None):
      # For the "data" vampprior the pseudo inputs are fixed over all epochs
      # and batches.
      return self.pseudo_inputs



# utility functions 

def inh_factor_b(true_cov,precon_mat,debug = False):
  import tensorflow as tf 
  d = true_cov.shape[0]
  precon_mat = tf.cast(precon_mat, tf.float32)
  true_cov = tf.cast(true_cov,tf.float32)
  proposal_cov = tf.linalg.matmul(precon_mat, tf.transpose(precon_mat, perm = [0,2,1]))

  SS_inv = tf.linalg.matmul(true_cov,tf.linalg.inv(proposal_cov))

  eigvals, __= tf.linalg.eigh(SS_inv)
  eig_sqr = tf.math.sqrt(eigvals)
  sum_eig_sqr = tf.math.reduce_sum(eig_sqr, axis= -1) ** 2
  sum_eigen = tf.math.reduce_sum(eigvals, axis= -1)
  b = d * sum_eigen * (1 / sum_eig_sqr)

  if debug == True:
  	tf.print("d :", d)
  	tf.print("covariances dot",SS_inv)
  	tf.print("eigenvalues:", eigvals)
  	tf.print("true cov :",true_cov)
  	tf.print("proposal cov:",proposal_cov)
  	tf.print("eig_sqr",eig_sqr)
  	tf.print("sum_eig_sqr",sum_eig_sqr)
  	tf.print("sum_eigen", sum_eigen)


  return b

def scale_images(images, new_shape):
  from tqdm import tqdm
  from numpy import asarray
  from skimage.transform import resize
  images_list = list()
  for image in tqdm(images):
    # resize with nearest neighbor interpolation
    new_image = resize(image, new_shape, 0)
    # store
    images_list.append(new_image)
  return asarray(images_list)


def KID_score(real_images, synthetic_images, batch_size = 8, samples = 2000):
  import tensorflow as tf
  import numpy as np

  def image_resize(images):
    if images.shape[-1] is 1:
      images = tf.image.grayscale_to_rgb(images)
    images = tf.image.resize(
            images,
            size=[150, 150],
            method="bicubic",
            antialias=True,
        )
    images = tf.clip_by_value(images, 0.0, 1.0)
    images = tf.keras.applications.inception_v3.preprocess_input(images * 255.0)
    return images

  def polynomial_kernel(features_1, features_2):
    feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
    return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0


  def compute_kid(real_images, generated_images, batch_size=2, samples = 2000):

    model = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
            keras.applications.InceptionV3(include_top=False,input_shape=(150, 150, 3),weights="imagenet"),
                tf.keras.layers.GlobalAveragePooling2D()])
    #model.summary()

    real_images = tf.random.shuffle(real_images)
    generated_images = tf.random.shuffle(generated_images)

    real_images = image_resize(real_images[:samples])
    generated_images = image_resize(generated_images[:samples])

    real_features = model.predict(real_images, batch_size = batch_size, verbose = 0)
    generated_features = model.predict(generated_images, batch_size = batch_size, verbose = 0)

    kernel_real = polynomial_kernel(real_features, real_features)
    kernel_generated = polynomial_kernel(
        generated_features, generated_features
    )
    kernel_cross = polynomial_kernel(real_features, generated_features)

    # estimate the squared maximum mean discrepancy using the average kernel values
    batch_size = tf.shape(real_features)[0]
    batch_size_f = tf.cast(batch_size, dtype=tf.float32)
    mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
        batch_size_f * (batch_size_f - 1.0)
    )
    mean_kernel_generated = tf.reduce_sum(
        kernel_generated * (1.0 - tf.eye(batch_size))
    ) / (batch_size_f * (batch_size_f - 1.0))
    mean_kernel_cross = tf.reduce_mean(kernel_cross)
    kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross


    return kid

  kid = compute_kid(real_images, synthetic_images, batch_size, samples = samples)
  #print('KID score', np.array(kid))
  return np.array(kid)

def FID_score(real_images, synthetic_images, samples = 5000):
  
  import numpy
  from numpy import cov
  from numpy import trace
  from numpy import iscomplexobj
  from numpy import asarray
  from numpy.random import shuffle
  from scipy.linalg import sqrtm
  from keras.applications.inception_v3 import InceptionV3
  from keras.applications.inception_v3 import preprocess_input
  from keras.datasets.mnist import load_data
  from skimage.transform import resize
  from tqdm import tqdm

  def scale_images(images, new_shape):
    images_list = list()
    for image in tqdm(images):
      # resize with nearest neighbor interpolation
      new_image = resize(image, new_shape, 0)
      # store
      images_list.append(new_image)
    return asarray(images_list)
  
  # calculate frechet inception distance
  def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1, batch_size = 8)
    act2 = model.predict(images2, batch_size = 8)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  # prepare the inception v3 model
  model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
  
  images1 = real_images[:samples].astype('float32')
  images2 = synthetic_images[:samples].astype('float32')
  # resize images
  images1 = scale_images(images1, (299,299,3))
  images2 = scale_images(images2, (299,299,3))
  # pre-process images
  images1 = preprocess_input(images1)
  images2 = preprocess_input(images2)
  # calculate fid
  fid = calculate_fid(model, images1, images2)
  #print('FID: %.3f' % fid)
  return fid

def preprocess_oasis(images, normalize = True, binarization = 'static'):
  if normalize == True:
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2], 1)) / 255.
  else:
    # used in omniglot
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2], 1))

  if binarization == 'static':
    return np.where(images > .5, 1.0, 0.0).astype('float32')

  if binarization == 'dynamic':
    images = np.random.binomial(1, images)
    return images.astype('float32')

def preprocess_binary_images(images, normalize = True, binarization = 'static'):
  if normalize == True:
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1)) / 255.
  else:
    # used in omniglot
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

  if binarization == 'static':
    return np.where(images > .5, 1.0, 0.0).astype('float32')

  if binarization == 'dynamic':
    images = np.random.binomial(1, images)
    return images.astype('float32')

def preprocess_images_logistic_mixture(images):
  images = (images  - 127.5) / 127.5
  images = images.reshape((images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
  return images.astype('float32')

def preprocess_images(images, normalize = True):
      #images = (images - 127.5) / 127.5
      #images = images / 255.
  if normalize == True:
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1)) / 255.
  else:
    # used in omniglot
    images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

  return images.astype('float32')

def preprocess_images_log_normal(images):
      #images = (images - 127.5) / 127.5
      #images = images / 255.
  images = images.reshape((images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
  return images.astype('float32')


def unwrap(x):
	return tf.compat.v1.flags.tf_decorator.unwrap(x)


def condition_number(precon_mat, true_cov):
  import numpy as np
  import tensorflow as tf
  true_cov = tf.cast(true_cov,tf.float32)
  precon_mat = tf.cast(precon_mat, tf.float32)
  
  #b_tSb = tf.linalg.matmul(precon_mat,tf.linalg.matmul(tf.linalg.inv(true_cov),precon_mat), transpose_a=True)
  b_tSb = tf.linalg.matmul(precon_mat,tf.linalg.matmul(true_cov,precon_mat), transpose_a=True)
  eigvals,__ = tf.linalg.eigh(b_tSb)
  condition_num = tf.math.reduce_max(eigvals, axis=-1) / tf.math.reduce_min(eigvals, axis=-1)

  return condition_num

def compute_approx_posterior_samples(model, data, samples):
  import numpy as np
  import tensorflow as tf
  mean_z, logvar_z = model.encode(data)
  eps = tf.random.normal(shape = [int(samples)] + mean_z.shape)

  expanded_x_test = tf.tile(data[tf.newaxis, :], [int(samples)] + [1 for i in range(len(data.shape))])

  x = tf.reshape(expanded_x_test, [-1] + data.shape[1:].as_list())
  z0 = tf.reshape(model.reparameterize_initial_sample(mean_z, logvar_z, eps), [-1, eps.shape[-1]])
  z_trace, __, __ = model.apply_MCMC_kernels(z0, model.target_log_prob_fn(x), x)

  return z_trace[-1]

def generate_data_samples(model,data, samples):
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  mean_z, logvar_z = model.encode(data)

  eps = tf.random.normal(shape = [int(samples)] + mean_z.shape)

  expanded_x_test = tf.tile(data[tf.newaxis, :], [int(samples)] + [1 for i in range(len(data.shape))])

  x = tf.reshape(expanded_x_test, [-1] + data.shape[1:].as_list())
  z0 = tf.reshape(model.reparameterize_initial_sample(mean_z, logvar_z, eps), [-1, eps.shape[-1]])
  z_trace, __, __ = model.apply_MCMC_kernels(z0, model.target_log_prob_fn(x), x)

  mean_x_final, log_var_x_final = model.decode(z_trace[-1])

  samples = tfd.Normal(mean_x_final,log_var_x_final).sample()

  return samples


def corelated_covar(cor, latent_dim):
  import numpy as np
  import tensorflow as tf
  acc  = []
  for i in range(latent_dim):
      row = np.ones((1,latent_dim)) * cor
      row[0][i] = 1
      acc.append(row)
  
  cov = np.concatenate(acc,axis=0)
  C = np.linalg.cholesky(cov)

  correlated_cov = np.matmul(C, C.T)
  correlated_cov = tf.cast(correlated_cov, dtype = tf.float32)

  return correlated_cov

def generate_and_save_images(model, epoch, test_sample):

    mean_z, logvar_z = model.encode(test_sample)
    z, kernel_results = model.reparameterize(
      mean_z, logvar_z,
      target_log_prob_fn = model.target_log_prob_fn(test_sample),
      x = test_sample)

    predictions = model.decode(z, True)

    fig = plt.figure(figsize = (4, 4))
    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i + 1)
      plt.imshow(predictions[i, :, :, 0], cmap = 'gray')
      plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(os.path.join(path,'image_at_epoch_{:04d}.png'.format(epoch)))

def Callback_EarlyStopping(LossList, min_delta=0.1, patience=20):
    import numpy as np
   
    if len(LossList)//patience < 2 :
        return False
    

    mean_previous = np.mean(LossList[::-1][patience:2*patience]) 
    mean_recent = np.mean(LossList[::-1][:patience])
    
    delta_abs = np.abs(mean_recent - mean_previous) 
    delta_abs = np.abs(delta_abs / mean_previous) 
    if delta_abs < min_delta :
        print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
        print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False


def logistic_mixture_log_likelihood(x, x_decoded_m, x_decoded_invs, x_logit_weights, mixture_components, img_rows, img_cols, img_chns):
  import tensorflow as tf
  # Repeat the target to match the number of mixture component shapes

  x = tf.reshape(x, (-1, img_rows, img_cols, img_chns))
  slices = []
  for c in range(img_chns):
      slices += [x[:, :, :, c:c+1]] * (mixture_components)
  x = tf.concat(slices, axis=-1)

  # Pixels rescaled to be in [-1, 1] interval
  # print(x.shape)
  # print(x_decoded_m.shape)

  offset = 1. / 127.5 / 2.
  centered_mean = x - x_decoded_m
  

  cdfminus_arg = (centered_mean - offset) * tf.math.exp(x_decoded_invs)
  cdfplus_arg = (centered_mean + offset) * tf.math.exp(x_decoded_invs)

  cdfminus_safe = tf.sigmoid(cdfminus_arg)
  cdfplus_safe = tf.sigmoid(cdfplus_arg)

  # logistic pdf
  mid_in = centered_mean * tf.math.exp(x_decoded_invs)
  log_pdf_mid = -mid_in - x_decoded_invs - 2. * tf.math.softplus(-mid_in)

  # edge case treatment
  edge_case = log_pdf_mid - tf.math.log(127.5) + 2.04 * x_decoded_invs - 0.107

  log_cdfplus = cdfplus_arg - tf.math.softplus(cdfplus_arg)
  log_1minus_cdf = -tf.math.softplus(cdfminus_arg)

  log_ll = tf.where(x <= -0.999, log_cdfplus,
                        tf.where(x >= 0.999, log_1minus_cdf,
                                   tf.where(cdfplus_safe - cdfminus_safe > 1e-5,
                                              tf.math.log(tf.math.maximum(cdfplus_safe - cdfminus_safe, 1e-12)),
                                              edge_case)))


  pre_result = tf.nn.log_softmax(x_logit_weights, axis = -1) + log_ll

  result = []
  for chn in range(img_chns):
      chn_result = pre_result[:, :, :, chn*mixture_components:(chn+1)*mixture_components]
      v = tf.math.reduce_logsumexp(chn_result, axis=-1)
      result.append(v)

  result = tf.keras.backend.batch_flatten(tf.concat(result, axis=-1))

  return tf.reduce_sum(result, axis = -1)


def pixelcnn_loss(target, output_m,output_invs,output_logit_weights, img_rows, img_cols, img_chns, n_components):
    ''' Keras PixelCNN loss function. Use a lambda to fill in the last few
        parameters
        Args:
            img_rows, img_cols, img_chns: image dimensions
            n_components: number of mixture components
        Returns:
            log-loss
    '''
    def logsoftmax(x):
      ''' Numerically stable log(softmax(x)) '''
      m = K.max(x, axis=-1, keepdims=True)
      return x - m - K.log(K.sum(K.exp(x - m), axis=-1, keepdims=True))


    import tensorflow as tf
    import numpy as np
    K = tf.keras.backend
    # Extract out each of the mixture parameters (multiple of 3 b/c of image channels)

    # Repeat the target to match the number of mixture component shapes
    x = K.reshape(target, (-1, img_rows, img_cols, img_chns))
    slices = []
    for c in range(img_chns):
        slices += [x[:, :, :, c:c+1]] * n_components
    x = K.concatenate(slices, axis=-1)

    x_decoded_m = output_m
    x_decoded_invs = -output_invs
    x_logit_weights = output_logit_weights

    # Pixels rescaled to be in [-1, 1] interval
    offset = 1. / 127.5 / 2.
    centered_mean = x - x_decoded_m

    cdfminus_arg = (centered_mean - offset) * K.exp(x_decoded_invs)
    cdfplus_arg = (centered_mean + offset) * K.exp(x_decoded_invs)

    cdfminus_safe = K.sigmoid(cdfminus_arg)
    cdfplus_safe = K.sigmoid(cdfplus_arg)

    # Generate the PDF (logistic) in case the `m` is way off (cdf is too small)
    # pdf = e^(-(x-m)/s) / {s(1 + e^{-(x-m)/s})^2}
    # logpdf = -(x-m)/s - log s - 2 * log(1 + e^(-(x-m)/s))
    #        = -mid_in - invs - 2 * softplus(-mid_in)
    mid_in = centered_mean * K.exp(x_decoded_invs)
    log_pdf_mid = -mid_in - x_decoded_invs - 2. * tf.math.softplus(-mid_in)

    # Use trick from PixelCNN++ implementation to protect against edge/overflow cases
    # In extreme cases (cdfplus_safe - cdf_minus_safe < 1e-5), use the
    # log_pdf_mid and assume that density is 1 pixel width wide (1/127.5) as
    # the density: log(pdf * 1/127.5) = log(pdf) - log(127.5)
    # Add on line of best fit (see notebooks/blog post) to the difference between
    # edge case and the standard case
    edge_case = log_pdf_mid - np.log(127.5) + 2.04 * x_decoded_invs - 0.107

    # ln (sigmoid(x)) = x - ln(e^x + 1) = x - softplus(x)
    # ln (1 - sigmoid(x)) = ln(1 / (1 + e^x)) = -softplus(x)
    log_cdfplus = cdfplus_arg - tf.math.softplus(cdfplus_arg)
    log_1minus_cdf = -tf.math.softplus(cdfminus_arg)
    log_ll = tf.where(x <= -0.999, log_cdfplus,
                        tf.where(x >= 0.999, log_1minus_cdf,
                                   tf.where(cdfplus_safe - cdfminus_safe > 1e-5,
                                              K.log(K.maximum(cdfplus_safe - cdfminus_safe, 1e-12)),
                                              edge_case)))

    # x_weights * [sigma(x+0.5...) - sigma(x-0.5 ...) ]
    # = log x_weights + log (...)
    # Compute log(softmax(.)) directly here, instead of doing 2-step to avoid overflow
    pre_result = logsoftmax(x_logit_weights) + log_ll

    result = []
    for chn in range(img_chns):
        chn_result = pre_result[:, :, :, chn*n_components:(chn+1)*n_components]
        v = tf.math.reduce_logsumexp(chn_result, axis=-1)
        result.append(v)
    result = K.batch_flatten(K.stack(result, axis=-1))

    return K.sum(result, axis=-1)


# def gen_image(model, num_samples=batch_size):
#     x_sample = np.zeros((num_samples, img_rows, img_cols, img_chns))
    
#     # Iteratively generate each conditional pixel P(x_i | x_{1,..,i-1})
#     for i in range(img_rows):
#         for j in range(img_cols):
#             for k in range(img_chns):
#                 # =======================================================
#                 #x_out = model.predict(X_train, num_samples)
#                 x_out = model.predict(x_sample, num_samples)
#                 for n in range(num_samples):
#                     offset = k * mixture_components
#                     x_ms = x_out[n, i, j, offset:offset + mixture_components]
                    
#                     offset = mixture_components * img_chns + k * mixture_components
#                     x_invs = x_out[n, i, j, offset:offset + mixture_components]
                    
#                     offset = 2 * mixture_components * img_chns + k * mixture_components
#                     weights = softmax(x_out[n, i, j, offset:offset + mixture_components])
                    
#                     pvals = compute_mixture(x_ms, x_invs, weights, mixture_components)
#                     pvals /= (np.sum(pvals) + 1e-5)
#                     pixel_val = np.argmax(np.random.multinomial(1, pvals))
#                     x_sample[n, i, j, k] = (pixel_val - 127.5) / 127.5
        
                
#     return (x_sample * 127.5 + 127.5)


def compute_mixture(ms, invs, weights, n_comps):
  import math
  import numpy as np
  import tensorflow as tf


  def logistic_cdf(x, loc, scale):
      return tf.sigmoid((x - loc) / scale)


  def compute_pvals(m, invs):
      pvals = []
      for i in range(256):
          if i == 0:
              pval = logistic_cdf((0.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs))
          elif i == 255:
              pval = 1. - logistic_cdf((254.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs))
          else:
              pval = (logistic_cdf((i + 0.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs))
                      - logistic_cdf((i - 0.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs)))
          pvals.append(pval)

      return pvals


  components = []
  for i in range(n_comps):
      pvals = compute_pvals(ms[i], invs[i])
      arr = np.array(pvals)
      components.append(weights[i] * arr)
  return np.sum(components, axis=0)


  

  def discretized_mix_logistic_loss__(x,l,sum_all=True):

    def int_shape(x):
      return list(map(int, x.get_shape()))

    def concat_elu(x):
        """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
        axis = len(x.get_shape())-1
        return tf.nn.elu(tf.concat([x, -x], axis))

    def log_sum_exp(x):
        """ numerically stable log_sum_exp implementation that prevents overflow """
        axis = len(x.get_shape())-1
        m = tf.reduce_max(x, axis)
        m2 = tf.reduce_max(x, axis, keepdims=True)
        return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

    def log_prob_from_logits(x):
        """ numerically stable log_softmax implementation that prevents overflow """
        axis = len(x.get_shape())-1
        m = tf.reduce_max(x, axis, keepdims=True)
        return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))

    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic(xs,l,nr_mix, t = 1.0):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.shape, minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)

    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) / t *(tf.log(u) - tf.log(1. - u))

    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)

    print(x0)
    print(x1)
    print(x2)

    return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tqdm import tqdm


# The first layer is the PixelCNN layer. This layer simply
# builds on the 2D convolutional layer, but includes masking.
class PixelConvLayer(layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


# Next, we build our residual block layer.
# This is just a normal residual block, but based on the PixelConvLayer.
class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )
        self.pixel_conv = PixelConvLayer(
            mask_type="B",
            filters=filters // 2,
            kernel_size=3,
            activation="relu",
            padding="same",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return keras.layers.add([inputs, x])

def gen_image(model,epoch,test_sample, mixture_components):
    num_samples, img_rows, img_cols, img_chns = test_sample.shape
    num_samples = 16

    x_sample = np.zeros((num_samples, img_rows, img_cols, img_chns))
    
    # Iteratively generate each conditional pixel P(x_i | x_{1,..,i-1})
    for i in range(img_rows):
        for j in range(img_cols):
            for k in range(img_chns):
                mean_z, logvar_z = model.encode(test_sample)
                z, kernel_results = model.reparameterize(
                  mean_z, logvar_z,
                  target_log_prob_fn = model.target_log_prob_fn(test_sample),
                  x = test_sample)

                x_out = np.array(model.decoder(z))
                #print(x_out.shape)
                
                for n in range(num_samples):
                    offset = k * mixture_components
                    x_ms = x_out[n, i, j, offset:offset + mixture_components]
                    
                    offset = mixture_components * img_chns + k * mixture_components
                    x_invs = x_out[n, i, j, offset:offset + mixture_components]
                    
                    offset = 2 * mixture_components * img_chns + k * mixture_components
                    weights = tf.nn.softmax(x_out[n, i, j, offset:offset + mixture_components])
                    
                    pvals = compute_mixture(x_ms, x_invs, weights, mixture_components)
                    
                    pvals /= (np.sum(pvals) + 1e-5)
                    #print(pvals.shape)
                    pixel_val = np.argmax(np.random.multinomial(1, pvals))

                    x_sample[n, i, j, k] = (pixel_val - 127.5) / 127.5

    fig = plt.figure(figsize = (4, 4))
    for i in range(num_samples):
      plt.subplot(4, 4, i + 1)
      plt.imshow(x_sample[i, :, :, 0], cmap = 'gray')
      plt.axis('off')

    plt.savefig(os.path.join(path,'image_at_epoch_{:04d}.png'.format(epoch)))


def discretized_mix_logistic_loss_(x,l,sum_all=True):

    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    def int_shape(x):
      return list(map(int, x.get_shape()))

    def log_prob_from_logits(x):
      """ numerically stable log_softmax implementation that prevents overflow """
      axis = len(x.get_shape())-1
      m = tf.reduce_max(x, axis, keepdims=True)
      return x - m - tf.math.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))


    def log_sum_exp(x):
      """ numerically stable log_sum_exp implementation that prevents overflow """
      axis = len(x.get_shape())-1
      m = tf.reduce_max(x, axis)
      m2 = tf.reduce_max(x, axis, keepdims=True)
      return m + tf.math.log(tf.reduce_sum(tf.exp(x-m2), axis))

    xs = x.shape # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = l.shape # predicted distribution, e.g. (B,32,32,100)

    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic__(l,nr_mix):
    
    def int_shape(x):
          return list(map(int, x.get_shape()))

    ls = int_shape(l)

    #print(l.shape)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)

    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))

    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)


    return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3) 



import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import mixture_same_family
from tensorflow_probability.python.distributions import quantized_distribution
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.layers import weight_norm


class logistic_mixture_distribution():

  

  def __init__(self,
               image_shape,
               num_logistic_mix=5,
               high=255.,
               low=0.,
               dtype=tf.float32,
                name='logistic_mixture_distribution'):

    super(logistic_mixture_distribution, self).__init__()

    self._high = high
    self._low = low
    self._num_logistic_mix = num_logistic_mix


    # image_shape = tensorshape_util.constant_value_as_shape(image_shape)
    # image_input_shape = tensorshape_util.concatenate([None], image_shape)
    # input_shape = image_input_shape
    self.image_shape = image_shape



  def _make_mixture_dist(self, component_logits, locs, scales):
    """Builds a mixture of quantized logistic distributions.
    Args:
      component_logits: 4D `Tensor` of logits for the Categorical distribution
        over Quantized Logistic mixture components. Dimensions are `[batch_size,
        height, width, num_logistic_mix]`.
      locs: 4D `Tensor` of location parameters for the Quantized Logistic
        mixture components. Dimensions are `[batch_size, height, width,
        num_logistic_mix, num_channels]`.
      scales: 4D `Tensor` of location parameters for the Quantized Logistic
        mixture components. Dimensions are `[batch_size, height, width,
        num_logistic_mix, num_channels]`.
    Returns:
      dist: A quantized logistic mixture `tfp.distribution` over the input data.
    """
    mixture_distribution = tfd.Categorical(logits=component_logits)

    # Convert distribution parameters for pixel values in
    # `[self._low, self._high]` for use with `QuantizedDistribution`
    locs = self._low + 0.5 * (self._high - self._low) * (locs + 1.)
    scales *= 0.5 * (self._high - self._low)
    logistic_dist = tfd.QuantizedDistribution(
        distribution=tfd.TransformedDistribution(
            distribution=tfd.Logistic(loc=locs, scale=scales),
            bijector=tfb.Shift(shift=tf.cast(-0.5, self.dtype))),
        low=self._low, high=self._high)

    dist = tfd.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=tfd.Independent(
            logistic_dist, reinterpreted_batch_ndims=1))


    return tfd.Independent(dist, reinterpreted_batch_ndims=2)

def _log_prob(self, value, component_logits, locs, scales, coeffs):
    """Log probability function with optional conditional input.
    Calculates the log probability of a batch of data under the modeled
    distribution (or conditional distribution, if conditional input is
    provided).
    Args:
      value: `Tensor` or Numpy array of image data. May have leading batch
        dimension(s), which must broadcast to the leading batch dimensions of
        `conditional_input`.
      conditional_input: `Tensor` on which to condition the distribution (e.g.
        class labels), or `None`. May have leading batch dimension(s), which
        must broadcast to the leading batch dimensions of `value`.
      training: `bool` or `None`. If `bool`, it controls the dropout layer,
        where `True` implies dropout is active. If `None`, it defaults to
        `tf.keras.backend.learning_phase()`.
    Returns:
      log_prob_values: `Tensor`.
    """
    # Determine the batch shape of the input images
    image_batch_shape = prefer_static.shape(value)[:-3]

    # Broadcast `value` and `conditional_input` to the same batch_shape
    
    image_batch_and_conditional_shape = image_batch_shape

    value = tf.reshape(
        value, prefer_static.concat([(-1,), self.event_shape], axis=0))


    num_channels = self.event_shape[-1]

    if num_channels == 1:
      component_logits, locs, scales = params
    else:
      # If there is more than one channel, we create a linear autoregressive
      # dependency among the location parameters of the channels of a single
      # pixel (the scale parameters within a pixel are independent). For a pixel
      # with R/G/B channels, the `r`, `g`, and `b` saturation values are
      # distributed as:
      #
      # r ~ Logistic(loc_r, scale_r)
      # g ~ Logistic(coef_rg * r + loc_g, scale_g)
      # b ~ Logistic(coef_rb * r + coef_gb * g + loc_b, scale_b)
      # TODO(emilyaf) Investigate using fill_triangular/matrix multiplication
      # on the coefficients instead of split/multiply/concat
      component_logits, locs, scales, coeffs = params

      num_coeffs = num_channels * (num_channels - 1) // 2
      loc_tensors = tf.split(locs, num_channels, axis=-1)
      coef_tensors = tf.split(coeffs, num_coeffs, axis=-1)
      channel_tensors = tf.split(transformed_value, num_channels, axis=-1)

      coef_count = 0
      for i in range(num_channels):
        channel_tensors[i] = channel_tensors[i][..., tf.newaxis, :]
        for j in range(i):
          loc_tensors[i] += channel_tensors[j] * coef_tensors[coef_count]
          coef_count += 1
      locs = tf.concat(loc_tensors, axis=-1)

    dist = self._make_mixture_dist(component_logits, locs, scales)

    return tf.reshape(dist.log_prob(value), image_batch_and_conditional_shape)



def _sample_n(self, n, seed=None, conditional_input=None, training=False):
    """Samples from the distribution, with optional conditional input.
    Args:
      n: `int`, number of samples desired.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      conditional_input: `Tensor` on which to condition the distribution (e.g.
        class labels), or `None`.
      training: `bool` or `None`. If `bool`, it controls the dropout layer,
        where `True` implies dropout is active. If `None`, it defers to Keras'
        handling of train/eval status.
    Returns:
      samples: a `Tensor` of shape `[n, height, width, num_channels]`.
    """
    if conditional_input is not None:
      conditional_input = tf.convert_to_tensor(
          conditional_input, dtype=self.dtype)
      conditional_event_rank = tensorshape_util.rank(self.conditional_shape)
      conditional_input_shape = prefer_static.shape(conditional_input)
      conditional_sample_rank = prefer_static.rank(
          conditional_input) - conditional_event_rank

      # If `conditional_input` has no sample dimensions, prepend a sample
      # dimension
      if conditional_sample_rank == 0:
        conditional_input = conditional_input[tf.newaxis, ...]
        conditional_sample_rank = 1

      # Assert that the conditional event shape in the `PixelCnnNetwork` is the
      # same as that implied by `conditional_input`.
      conditional_event_shape = conditional_input_shape[
          conditional_sample_rank:]
      with tf.control_dependencies([
          tf.assert_equal(self.conditional_shape, conditional_event_shape)]):

        conditional_sample_shape = conditional_input_shape[
            :conditional_sample_rank]
        repeat = n // prefer_static.reduce_prod(conditional_sample_shape)
        h = tf.reshape(
            conditional_input,
            prefer_static.concat([(-1,), self.conditional_shape], axis=0))
        h = tf.tile(h,
                    prefer_static.pad(
                        [repeat], paddings=[[0, conditional_event_rank]],
                        constant_values=1))

    samples_0 = tf.random.uniform(
        prefer_static.concat([(n,), self.event_shape], axis=0),
        minval=-1., maxval=1., dtype=self.dtype, seed=seed)
    inputs = samples_0 if conditional_input is None else [samples_0, h]
    params_0 = self.network(inputs, training=training)
    samples_0 = self._sample_channels(*params_0, seed=seed)

    image_height, image_width, _ = tensorshape_util.as_list(self.event_shape)


def discretized_mix_logistic_loss(x,l, debug = False):


    tf.Assert(tf.math.reduce_max(x) == 1.,[x])
    tf.Assert(tf.math.reduce_min(x) == -1.,[x])

    def int_shape(x):
      return list(map(int, x.get_shape()))

    def log_prob_from_logits(x):
      """ numerically stable log_softmax implementation that prevents overflow """
      axis = len(x.get_shape())-1
      m = tf.reduce_max(x, axis, keepdims=True)
      return x - m - tf.math.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))


    def log_sum_exp(x, axis = -1):
      """ numerically stable log_sum_exp implementation that prevents overflow """
      m = tf.reduce_max(x, axis)
      m2 = tf.reduce_max(x, axis, keepdims=True)
      return m + tf.math.log(tf.reduce_sum(tf.exp(x-m2), axis))

    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    

    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]

    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])


    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)

    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)  
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) 
    cdf_delta = cdf_plus - cdf_min 
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in)


    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.99, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    #log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    log_probs = tf.reduce_sum(log_probs,3) + tf.nn.log_softmax(logit_probs, axis = -1) 

    if debug == True:
      print('xs', xs)
      print('ls', ls)
      print('nr_mix', nr_mix)
      tf.print('logit_probs',logit_probs)
      tf.print('means', means)
      tf.print('log_scales', log_scales)
      tf.print('coeffs', coeffs)
    
    
    return tf.math.reduce_logsumexp(log_probs, axis = -1)
    
        

def sample_from_discretized_mix_logistic(l,nr_mix, t = 1.):

    def int_shape(x):
          return list(map(int, x.get_shape()))

    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    gumbel = - tf.math.log(-tf.math.log(tf.random.uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5)))
    sel = tf.one_hot(tf.argmax(logit_probs / t + gumbel, -1), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])

    #sel = tf.one_hot(tf.argmax(logit_probs / t - tf.math.log(-tf.math.log(tf.random.uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    #sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    #sel = 1.
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) / t * (tf.math.log(u) - tf.math.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)

    x = tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)
  
    

    x = x / 2. + 0.5
    

    return x

def predict_from_discretized_mix_logistic(l,nr_mix):

    def int_shape(x):
          return list(map(int, x.get_shape()))

    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.nn.softmax(logit_probs, axis = -1)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    x = means
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)

    x = tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)
    

    x = x / 2. + 0.5

    return x


class LogistixMixture:

  def log_prob(x, params, num_channels = 3, num_mixtures = 5, nbits = 8):

    bits = 2. ** nbits
    scale_min, scale_max = [0. , 1.]

    bin_size = (scale_max - scale_min) / (bits - 1.)
    eps = 1e-12

    # unpack parameters
    logit_probs = params[:,:,:,:num_mixtures]
    means = params[:,:,:,num_mixtures:(num_channels + 1) * num_mixtures]
    logscales = params[:,:,:,(num_channels + 1) * num_mixtures:(num_channels * 2 + 1) * num_mixtures]
    coeffs = params[:,:,:,(num_channels * 2 + 1) * num_mixtures:(num_channels * 2 + 4) * num_mixtures]

    # process parameters
    logit_probs = tf.nn.log_softmax(logit_probs, axis = -1)
    logscales = tf.clip_by_value(logscales, -7., np.inf)
    coeffs = tf.math.tanh(coeffs)

    mean0 = means[:, :, :, 0]
    mean1 = means[:,:,:,1] + coeffs[:,:,:,0] * x[:,:,:,0]
    mean2 = means[:,:,:,2] + coeffs[:,:,:,1] * x[:,:,:,0] + coeffs[:,:,:,2] * x[:,:,:,1]

    means = tf.stack([mean0, mean1, mean2], axis = -1)

    print('means_shape', means.shape)
    print('logscales_shape',logscales.shape)
    print('x_shape', x.shape)



    x_plus = tf.math.exp(-logscales) * (x - means + 0.5 * bin_size)
    x_minus = tf.math.exp(-logscales) * (x - means - 0.5 * bin_size)
    cdf_delta = tf.sigmoid(x_plus) - tf.sigmoid(x_minus)
    log_cdf_mid = tf.math.log(tf.clip_by_value(cdf_delta, clip_value_min = eps))

    upper = scale_max - 0.5 * bin_size 
    mask_upper = tf.where(x <= upper)
    log_cdf_up = - tf.math.softplus(x_minus)

    lower = scale_min + 0.5 * bin_size
    mask_lower  = tf.where(x >= lower)
    log_cdf_low = x_plus - tf.math.softplus(x_plus)


    x_in = tf.math.exp(-logscales) * (x - means)
    mask_delta = tf.where(cdf_delta > 1e-5)
    log_cdf_approx  = x_in - logscales - 2. * tf.math.softplus(x_in) + tf.math.log(bin_size)


    # Compute log CDF w/ extrime cases
    log_cdf = log_cdf_mid * mask_delta + log_cdf_approx * (1.0 - mask_delta)
    log_cdf = log_cdf_low * (1.0 - mask_lower) + log_cdf * mask_lower
    log_cdf = log_cdf_up  * (1.0 - mask_upper) + log_cdf * mask_upper


    log_prob = tf.math.reduce_logsumexp(tf.reduce_sum(log_cdf, axis = -1) + logit_probs, aixs = -1)

    return log_prob



def resnet50_encoder(input_shape, latent_dim):

  def res_identity(x, filters): 
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

    #first block 
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    #second block # bottleneck (but size kept same with padding)
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # third block activation used after adding the input
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input 
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x

  def res_conv(x, s, filters):
    '''
    here the input size changes''' 
    x_skip = x
    f1, f2 = filters

    # first block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # second block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    #third block
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut 
    x_skip = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    # add 
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x

  input_im = tf.keras.layers.Input(shape=(input_shape))
  x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(latent_dim + latent_dim, activation='linear')(x)

  # define the model 

  model = tf.keras.Model(inputs=input_im, outputs=x, name='Resnet50')

  return model



def resnet50_decoder(input_shape, output_dim):

  def res_identity(x, filters): 
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

    #first block 
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    #second block # bottleneck (but size kept same with padding)
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # third block activation used after adding the input
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input 
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x

  def res_conv(x, s, filters):
    '''
    here the input size changes''' 
    x_skip = x
    f1, f2 = filters

    # first block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # second block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    #third block
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut 
    x_skip = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    # add 
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x

  input_im = tf.keras.layers.Input(shape=(input_shape))
  x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

  x = res_conv(x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = res_conv(x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = res_conv(x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = res_conv(x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(output_dim, activation='linear')(x)
  x = tf.keras.layers.Reshape(32,32,50)

  # define the model 

  model = tf.keras.Model(inputs=input_im, outputs=x, name='Resnet50')

  return model




class ResidualEncoderCell:

  def __init__(self,input_shape = (32,32,3),downsample = False, se_ratio = 16, bn_momentum = 16, gamma_reg=None, use_bias = True, res_scalar = 0.1):

    super().__init__()

    #self.x = x
    self.se_ratio = se_ratio
    self.bn_momentum = bn_momentum
    self.gamma_reg = gamma_reg
    self.use_bias = use_bias
    self.downsample = downsample
    self.res_scalar = res_scalar
    self.input_shape = input_shape

  def SqueezeExciteLayer(self, x, ratio):

    size = x.shape[-1]
    squeeze_size = max(size // ratio, 4)

    pool = tf.keras.layers.GlobalAveragePooling2D()
    dense1 = tf.keras.layers.Dense(squeeze_size, activation = 'relu')
    dense2 = tf.keras.layers.Dense(size, activation = 'sigmoid')

    se_layer = tf.keras.Sequential([
      pool(),
      dense1(),
      dense2(),
      tf.keras.layers.Reshape(target_shape = (-1, 1, 1, x.shape[-1])),
    ])

    return se_layer * x

  def FactorizedDownsample(self,x, channels_out = None):


    channels_in = self.input_shape[-1]
    if channels_out is None:
      channels_out = channels_in * 2

    quarter = channels_out // 4
    lastqrt = channels_out - 3 * quarter
    self.conv1 =tf.keras.layers.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
    self.conv2 = tf.keras.layers.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
    self.conv3 = tf.keras.layers.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
    self.conv4 = tf.keras.layers.Conv2D(filters=lastqrt, kernel_size=(1, 1), strides=(2, 2), padding='same')

    stack1 = self.conv1(x)
    stack2 = self.conv2(x)
    stack3 = self.conv3(x)
    stack4 = self.conv4(x)

    down_sample_layer = tf.keras.Sequential([tf.keras.layers.concatenate([stack1,stack2,stack3,stack4], axis = -1)
    ])

    return down_sample_layer

  def ConvBlock(self,
                 kernel_size,
                 scale_channels=1,
                 downsample=False,
                 upsample=False,
                 depthwise=False,
                 use_bias=True,
                 weight_norm=False,
                 spectral_norm=True,
                 dilation_rate=1,
                 activation='linear',
                 padding='same',
                 abs_channels=None):

    upsample_layer = None
    if upsample:
      upsample_layer = tf.keras.layers.UpSampling2D((2,2), interpolation='nearest')

    channels_in = self.input_shape[-1]

    if abs_channels is None:
      assert scale_channels != 0
      if scale_channels > 0:
        abs_channels = channels_in * scale_channels
      else:
        assert channels_in % abs(scale_channels) == 0
        abs_channels = channels_in // abs(scale_channels)

    layer = tf.keras.layers.Conv2D(filters = abs_channels, kernal_size = kernel_size, strides = 1 if not downsample else 2, groups = 1 if not depthwise else abs_channels,
      use_bias = use_bias , dilation_rate = dilation_rate, activation = activation, padding = padding)

    layer_1x1_depth = None

    if depthwise:
      layer_1x1_depth = tf.keras.layers.Conv2D(filters = abs_channels, kernel_size = (1,1))

    if weight_norm:
      layer = tfa.layers.WeightNormalization(layer)
      if weight_norm and conv_depth1x1:
        layer_1x1_depth = tfa.layers.WeightNormalization(layer_1x1_depth)

    if spectral_norm:
      layer = tfa.layers.SpectralNormalization(layer)
      if spectral_norm and conv_depth1x1:
        layer_1x1_depth = tfa.layers.SpectralNormalization(layer_1x1_depth)

    if upsample:
      x = upsample_layer(x)

    if depthwise:

      return layer_1x1_depth(x)

    else:

      return layer(x)


  def define(self):


    # define costume layers
    self.bn0 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum, gamma_regularizer=self.gamma_reg)
    self.bn1 = tf.keras.layers.BatchNormalization(momentum=self.bn_momentum, gamma_regularizer=self.gamma_reg)

    if self.downsample:
        self.conv_3x3s_0 = self.ConvBlock(kernel_size=(3, 3), downsample=True, scale_channels=2)
        self.conv_3x3s_1 = self.ConvBlock(kernel_size=(3, 3))
    else:
        self.conv_3x3s_0 = self.ConvBlock(kernel_size=(3, 3))
        self.conv_3x3s_1 = self.ConvBlock(kernel_size=(3, 3))

    self.se_layer = self.SqueezeExciteLayer(self.se_ratio)
    
    if self.downsample:
        self.downsample_layer = self.FactorizedDownsample()


    encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = self.input_shape),
      self.bn0,
      tf.keras.activations.swish(),
      self.conv_3x3s_0,
      self.bn1,
      tf.keras.activations.swish(),
      self.conv_3x3s_1,
      self.se_layer
      ])

    return encoder



class ResidualDecoderCell:

  def __init__(self,
                 input_shape=(32,32,3),
                 upsample=False,
                 expand_ratio=6,
                 se_ratio=16,
                 bn_momentum=0.95,
                 gamma_reg=None,
                 use_bias=True,
                 res_scalar=0.1,
                 ):


    self.expand_ratio = expand_ratio
    self.se_ratio = se_ratio
    self.bn_momentum = bn_momentum
    self.gamma_reg = gamma_reg
    self.use_bias = use_bias
    self.upsample = upsample
    self.res_scalar = res_scalar
    
    self.conv_depthw = None
    self.conv_expand = None
    self.conv_reduce = None
    self.upsample_residual = None
    self.upsample_conv1x1 = None
    self.input_shape = input_shape

  def SqueezeExciteLayer(self, x, ratio):

    size = x.shape[-1]
    squeeze_size = max(size // ratio, 4)

    pool = tf.keras.layers.GlobalAveragePooling2D()
    dense1 = tf.keras.layers.Dense(squeeze_size, activation = 'relu')
    dense2 = tf.keras.layers.Dense(size, activation = 'sigmoid')

    se_layer = tf.keras.Sequential([
      pool(),
      dense1(),
      dense2(),
      tf.keras.layers.Reshape(target_shape = (-1, 1, 1, x.shape[-1])),
    ])

    return se_layer * x

  def FactorizedDownsample(self,x, channels_out = None):


    channels_in = self.input_shape[-1]
    if channels_out is None:
      channels_out = channels_in * 2

    quarter = channels_out // 4
    lastqrt = channels_out - 3 * quarter
    self.conv1 =tf.keras.layers.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
    self.conv2 = tf.keras.layers.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
    self.conv3 = tf.keras.layers.Conv2D(filters=quarter, kernel_size=(1, 1), strides=(2, 2), padding='same')
    self.conv4 = tf.keras.layers.Conv2D(filters=lastqrt, kernel_size=(1, 1), strides=(2, 2), padding='same')

    stack1 = self.conv1(x)
    stack2 = self.conv2(x)
    stack3 = self.conv3(x)
    stack4 = self.conv4(x)

    down_sample_layer = tf.keras.Sequential([tf.keras.layers.concatenate([stack1,stack2,stack3,stack4], axis = -1)
    ])

    return down_sample_layer

  def ConvBlock(self,
                 kernel_size,
                 scale_channels=1,
                 downsample=False,
                 upsample=False,
                 depthwise=False,
                 use_bias=True,
                 weight_norm=False,
                 spectral_norm=True,
                 dilation_rate=1,
                 activation='linear',
                 padding='same',
                 abs_channels=None):

    upsample_layer = None
    if upsample:
      upsample_layer = tf.keras.layers.UpSampling2D((2,2), interpolation='nearest')

    channels_in = self.input_shape[-1]

    if abs_channels is None:
      assert scale_channels != 0
      if scale_channels > 0:
        abs_channels = channels_in * scale_channels
      else:
        assert channels_in % abs(scale_channels) == 0
        abs_channels = channels_in // abs(scale_channels)

    layer = tf.keras.layers.Conv2D(filters = abs_channels, kernal_size = kernel_size, strides = 1 if not downsample else 2, groups = 1 if not depthwise else abs_channels,
      use_bias = use_bias , dilation_rate = dilation_rate, activation = activation, padding = padding)

    layer_1x1_depth = None

    if depthwise:
      layer_1x1_depth = tf.keras.layers.Conv2D(filters = abs_channels, kernel_size = (1,1))

    if weight_norm:
      layer = tfa.layers.WeightNormalization(layer)
      if weight_norm and conv_depth1x1:
        layer_1x1_depth = tfa.layers.WeightNormalization(layer_1x1_depth)

    if spectral_norm:
      layer = tfa.layers.SpectralNormalization(layer)
      if spectral_norm and conv_depth1x1:
        layer_1x1_depth = tfa.layers.SpectralNormalization(layer_1x1_depth)

    if upsample:
      x = upsample_layer(x)

    if depthwise:

      return layer_1x1_depth(x)

    else:

      return layer(x)


  def define(self):

    num_c = input_shape[-1]
    num_ec = num_c * self.expand_ratio

    self.bn_layers = [tf.keras.layers.BatchNormalization(
            momentum=self.bn_momentum,
            gamma_regularizer=self.gamma_reg) for _ in range(4)]


    self.conv_expand = self.ConvBlock(kernel_size=(1, 1), scale_channels=self.expand_ratio)

    # Depthwise separable convolution, with possible upsample
    if self.upsample:
      self.conv_depthw = self.ConvBlock(kernel_size=(5, 5),
                                      depthwise=True,
                                      upsample=True,
                                      scale_channels=-2,
                                      use_bias=False,
                                      weight_norm=False)

      self.upsample_residual = self.ConvBlock(kernel_size=(1,1), upsample=True, scale_channels=-2)
    else:
      self.conv_depthw = self.ConvBlock(kernel_size=(5, 5),
                                      depthwise=True,
                                      use_bias=False,
                                      weight_norm=False)
        
    self.conv_reduce = self.ConvBlock(kernel_size=(1, 1),
                                  scale_channels=-self.expand_ratio,
                                  use_bias=False,
                                  weight_norm=False)

    self.se_layer = self.SqueezeExciteLayer(self.se_ratio)


    decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape = self.input_shape),
      self.bn_layers[0],
      self.conv_expand,
      self.bn_layers[1],
      tf.keras.activations.swish(),
      self.conv_depthw,
      self.bn_layers[2],
      tf.keras.activations.swish(),
      self.conv_reduce,
      self.bn_layers[3],
      self.se_layer,
      ])

    return decoder



























































