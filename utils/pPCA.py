import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np



class pPCA():

    def __init__(self,latent_dim,data_dim,sample_dim,true_variance = 1,prior_variance = 1):
        self.latent_dim = latent_dim
        self.true_variance = true_variance
        self.data_dim = data_dim
        self.sample_dim = sample_dim
        self.prior_variance = prior_variance

    def synthesize(self):

        w_true = tfd.Normal(tf.zeros([self.data_dim, self.latent_dim]), tf.ones([self.data_dim, self.latent_dim])).sample()

        # Independent prior
        z_true = tfd.Normal(tf.zeros([self.latent_dim, self.sample_dim]), self.prior_variance * tf.ones([self.latent_dim, self.sample_dim])).sample() 
        #z_true = tfd.Normal(tf.zeros([self.latent_dim, self.sample_dim]),cov).sample() 

        b = tf.ones([self.data_dim,self.sample_dim])
        x_true = tfd.Normal(tf.matmul(w_true, z_true)+b,self.true_variance).sample()

        return w_true, z_true, x_true

    def solve_parameters(self,x):
        x = tf.transpose(x)
        covariance = np.cov(x, rowvar = False)
        w, u = np.linalg.eigh(covariance)
        eigvals, eigvecs = w[::-1], u[:,::-1]
        missing_eigvals = eigvals[self.latent_dim:]
        sigma_sq_mle = missing_eigvals.sum() / (eigvals.shape[0] - self.latent_dim)
  
        active_eigvals = np.diag(eigvals[:self.latent_dim])
        active_components = eigvecs[:,:self.latent_dim]
  
        W_mle = active_components.dot((active_eigvals - sigma_sq_mle * np.eye(self.latent_dim)) ** 0.5)

        #check if W solved same as W true 

        return sigma_sq_mle, W_mle

    def solve_posterior_covar(self):
        w, z, x= self.synthesize()
        w = np.matrix(w)
        sigma_sq, W = self.solve_parameters(x)

        #M = (w.H).dot(w) + self.true_variance * np.eye(w.shape[1])
        M = np.matmul(w.H, w) + self.true_variance * np.eye(w.shape[1])
        S = self.true_variance * np.linalg.inv(M)

        return S

    def solve_log_likelihood(self):
        import scipy.stats as ss
        w, z, x= self.synthesize()
        w = np.array(w)
        sigma_sq, W = self.solve_parameters(x)
        x = tf.transpose(x)
        covariance = np.cov(x, rowvar = False)
        #mu = np.mean(x, axis=0)
        # for zero bias models the mean of the marginal likelihood of X is zero
        mu = np.zeros(x.shape[1])

        d = covariance.shape[0]
        C = w.dot(w.T) + np.eye(W.shape[0])

        ll = tf.reduce_mean(ss.multivariate_normal.logpdf(x, mu, C))
        return ll

    def solve_target_gradients(self, wrt = "z"):
        # gradient of logp(z|x) w.r.t z 
        w, z, x= self.synthesize()
        sigma_sq, W = self.solve_parameters(x)
        mu = np.mean(x, axis = 0)



        M = (W.T).dot(W) + sigma_sq * np.eye(W.shape[1])
        M_inv = np.linalg.inv(M)

        mean = np.linalg.multi_dot([M_inv, W.T, (x - mu)])
        S = sigma_sq * M_inv
        S_inv = np.linalg.inv(S)

        if wrt == "z":
            d_lp_dz = - S_inv.dot(z - mu)

            return d_lp_dz

        if wrt == "S":
            d_lp_dS = - 0.5 * (S_inv - np.linalg.multi_dot([S_inv, (z - mean), tf.transpose((z - mean)), S_inv]))

            return d_lp_dS

        if wrt == 'L':

            L = np.linalg.cholesky(S)
            z_shifted = z - mean

            print(z_shifted.shape)
            print(z.shape)
            print(mean)
            
            z_T_L = np.dot(tf.transpose(z_shifted),L)
            LL_T_inv = np.linalg.inv(np.dot(L,L.T))

            d_lp_dL = - (np.dot(z_shifted,z_T_L) + np.dot(LL_T_inv,L))

            return d_lp_dL

    def solve_posterior_mean(self,x):
        w, z, x= self.synthesize()
        w = np.array(w)
        sigma_sq, W = self.solve_parameters(x)

        M = (w.T).dot(w) + self.true_variance * np.eye(w.shape[1])

        true_mean = np.linalg.inv(M).dot(w.T).dot(x)

        return true_mean
            
























