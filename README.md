# Learning variational autoencoders via MCMC speed measures

Official tensorflow implementation of the paper Learning variational autoencoders via MCMC speed measures (https://arxiv.org/abs/2308.13731)


# Getting Started

For setting up an environment for testing and development we recommend using python 3.9 for being consistent with our own development.
After setting up and environment make sure to install required modules using pip or conda as follows,

```
pip install -r requirements.txt
```

# Experiment Replication

## Marginal Log Likelihood Estimation

The first section of the paper is concerned with testing our proposed entropy based sampler against other popular sampling methods 
for improving Variational inference in Variational Autoencoder models. To replicate the results presented in the first table of the paper, 
where the marginal log likelihood is used as a model metric run the following command for testing on the MNIST dataset using the Vanilla VAE model.

```
python test_vae_experiments.py -data_set mnist -likelihood Bernoulli -latent_dim 10 -prior Isotropic_Gaussian -eval_nll True -nll_particles 100000
```

This will replicate the data point for the seed id you provide with the -id argument. 
To recover the mean reported in the paper randomly peek 3 random seeds and run the process above.

Moreover, for testing using MCMC coupled models you can specify the command below for training a model with a proposed method using 10 mcmc steps and two leapfrog integrators. 

```
python test_vae_experiments.py -sampler gradHMC -num_MCMC_steps 10 -num_leapfrog_steps 2 -data_set mnist -likelihood Bernoulli -latent_dim 10 -prior Isotropic_Gaussian -eval_nll True -nll_particles 100000
```

Take note that the estimation process is computationally expensive you may need to utilize a gpu for testing or reduce the estimation batch size or even reduce the number of particles.
Reducing the number of particles will bias the estimate, but it will preserve the relative difference among the models tested. 


## Kernel Inception Distance Estimation

The follow section of the paper seeks to evaluate model performance with an alternative metric termed the Kernel Inception Distance (KID) 
which is computed on samples generated from the model. In order to replicate results for the MNIST datasets and compute the distance you can execute the following command

```
python test_vae_experiments.py -data_set mnist -likelihood Bernoulli -latent_dim 10 -prior Isotropic_Gaussian -eval_kid True -netw cnn
```

This will train a vanilla VAE using convolution as the network architectures of the encoder and the decoder and store the KID score of the converged model locally. Alternatively,
you can do the same process by altering the sampler arguments as in the previous section to replicate the results of the paper and notice the increased qualitative image generation using the proposed method.

## Low-Sample Training and Classification tasks on Oasis Brain Dataset

Next you can replicate our results in a real world dataset, namely the oasis brain f-MRI dataset. This will require you to first download the data locally. To do so follow this link for the open-access version of the dataset, https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset. 
Once you download the dataset to replicate our experiment make sure to bundle all demented images in a directory named AD, while renaming the non-demented category to CN, standing for control. This forms the classification problem we sought to improve upon by data augmentation using synthetic images. To replicate our experiment in the paper,
first execute the command above by re specifying the directory containing the OASIS dataset. Not that synthetic generation is done only in the demented category, which is the minority class.

```
python oasis_vae_experiments.py -netw mlp -likelihood logistic -obs_log_var -2. -save_generated_data True -oasis_class AD -eval_kid True
```

This will generate a series of npy files containing samples from the vanilla VAE model with size of 200, 500, 1000, 2000. Alternatively, you can also generate images by training the model with our proposed method by specifying the sampler, number of steps, and leapfrog integrators.
Also, the kid score for the generated images will be computed, which is also reported. To replicate the results in the paper also make sure to run the command with the oasis_class command set to C for computing kid score for the majority class of the control patients. 

Following the synthetic data generation you can actually perform the identical classification task as we did in the paper by executing the command below.

```
python classifier.py -augmentation synthetic_raw_mix -synthetic_paths <synthetic_data_path> -synthetic_size 2000
```

Make sure to replace <synthetic_data_path> with the path of the 2000 sample size generated data from the VAE model to get an accurate representation of the data in the paper.


## Linear Hierarchical Variational Auto-encoders

To replicate the simulation data results reported in the paper for using Hierarchical VAEs you can execute the following command.

```
python test_chvae_experiments.py -likelihood Normal -data simulation -latent_dims [50,100]
```

Make sure to change the latent_dims argument accordingly to get the full spectrum as those tested in the paper and again alter the sampler, num_MCMC_steps, and num_leapfrog_steps arguments to 
test additional models using our proposed method. 

## Hierarchical Variational Auto-encoders

Finally, to replicate the last section of the paper utilizing non-linearity in the structure of the hierarchical VAE run the following command to replicate the results on MNIST

```
python test_chvae_experiments.py -likelihood Bernoulli -data mnist -latent_dims [5,10]
```

Again, adjust the sampler arguments to get the results using the samplers tested. 
























