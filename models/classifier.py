
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
#tf.config.list_physical_devices('GPU')
import tensorflow_probability as tfp
import numpy as np
import time
import matplotlib.pyplot as plt
from absl import app
from absl import flags
import os
import sys
from VAE_model import VAE
import pandas as pd
#from load_oasis import load_oasis
from load_oasis_3 import load_oasis
from load_adni import load_adni
import cv2

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import matthews_corrcoef as MCC
import random
from utils import preprocess_binary_images, preprocess_images

from load_chest_xray import load_pneumonia


'''
VAE-id 0 latent dim = 2


VAE id 1 900-100 5 leaps 25 steps latent = 2



'''


flags.DEFINE_float("learning_rate",
                     default=0.0001,
                     help="learning rate for all optimizers")
flags.DEFINE_string("augmentation",
                     default="synthetic_raw_mix",
                     help="none or synthetic_raw_mix or synthetic or oversample")
flags.DEFINE_string("augmentation_regime",
                     default="all",
                     help="class specified data injection")
flags.DEFINE_integer("id",
                     default=1,
                     help="id of run")
flags.DEFINE_integer("epochs",
                     default=500,
                     help="training epochs")
flags.DEFINE_integer("synthetic_size",
                     default=0,
                     help="number of synthetic data")
flags.DEFINE_string("likelihood",
                     default="Bernoulli",
                     help="likelihood of the generator")
flags.DEFINE_string("model",
                     default="mlp",
                     help="classifier architecture")
flags.DEFINE_string("oasis_transform",
                     default="",
                     help="None or rHC or lHC")
flags.DEFINE_integer("oasis_slice_id",
                     default=0,
                     help="ranges from 0-")
flags.DEFINE_string("VAE_model",
                     default="VAE-dsHMC-only-AD",
                     help="VAE or some VAE-mcmc variant")
flags.DEFINE_list("synthetic_paths",
                     default=[
"C:/Users/kreou/OneDrive/Documents/GitHub/vae-main/adni_hope/AD/0/adni__dsHMC__5__True__0.001__0.01__0__logistic__Isotropic_Gaussian__dsHMC__mlp__2__True__20__0__-2.0__1.0__1.2__1__2000__1900"],
help="paths of synthetic images used for augmentation given in order of labels [0,...,k] k=class number")

flags.DEFINE_string(
    'data_set',
    default='adni',
    help="data set mnist or fashion_mnist or oasis")
flags.DEFINE_string(
    'model_dir',
    default=os.path.join(os.getcwd(),'classifier_adni_final'),
    help="Directory to put the model's fit and outputs.")
FLAGS = flags.FLAGS

def main(argv):
    del argv  # unused

  #save (command line) flags to file
    fv = flags._flagvalues.FlagValues()
    key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
    s = '\n'.join(f.serialize() for f in key_flags)
    print('specified flags:\n{}'.format(s))
    path=os.path.join(FLAGS.model_dir,flags.FLAGS.VAE_model,flags.FLAGS.oasis_transform,str(flags.FLAGS.oasis_slice_id),str(flags.FLAGS.augmentation),flags.FLAGS.model,str(flags.FLAGS.synthetic_size),str(FLAGS.id),
                    '{}__{}__{}__{}__{}'.format(
                      flags.FLAGS.augmentation,
                      flags.FLAGS.learning_rate,
                      flags.FLAGS.likelihood, 
                      flags.FLAGS.synthetic_size,
                      flags.FLAGS.model))
    if not os.path.exists(path):
        os.makedirs(path)
    flag_file = open(os.path.join(path,'flags.txt'), "w")
    flag_file.write(s)
    flag_file.close()


    #set seeds to id
    tf.random.set_seed(FLAGS.id)
    np.random.seed(FLAGS.id)


    # Data Processing
    if FLAGS.data_set == 'pneumonia':

        train_images, test_images, train_labels, test_labels = load_pneumonia(random_state =FLAGS.id)
        classes = 2
        CLASSES = ['Normal', 'Pneumonia']
        normalize = False
        binarization = 'static'

        print('test_images.shape', test_images.shape)
        print('train_images.shape', train_images.shape)

    if FLAGS.data_set == 'oasis':
        # normal_images, normal_labels, dementia_images, dementia_labels  = load_oasis()
        # classes = 2
        # CLASSES = ['normal', 'dementia']
        train_images, test_images, train_labels, test_labels = load_oasis(transform = FLAGS.oasis_transform, slice_id = FLAGS.oasis_slice_id, random_state =FLAGS.id)

        classes = 2
        CLASSES = ['Demented', 'NonDemented']
        # train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.33, random_state=FLAGS.id, stratify =labels)
        normalize = False
        binarization = 'static'

        # train_images = preprocess_binary_images(train_images, normalize = normalize, binarization = binarization)
        # test_images = preprocess_binary_images(test_images, normalize = normalize, binarization = binarization)
        print('test_images.shape', test_images.shape)
        print('train_images.shape', train_images.shape)

    if FLAGS.data_set == 'adni':
        #normal_images, normal_labels, dementia_images, dementia_labels = load_oasis(split_classes = True, colab = False)
        classes = 2
        train_images , test_images, train_labels, test_labels = load_adni(random_state = FLAGS.id)

        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)

        print('train_images',train_images.shape)
        print('test_images', test_images.shape)

        if classes == 4:
            CLASSES = [ 'NonDemented','VeryMildDemented','MildDemented','ModerateDemented']
        if classes == 2:
            CLASSES = ['NonDemented', 'Demented']

    if FLAGS.data_set == 'reduced_MNIST':
        
        train_images, test_images, train_labels, test_labels = load_reduced_mnist()
        classes = 10





    # train_images = tf.expand_dims(train_images, axis = -1)
    # test_images = tf.expand_dims(test_images, axis = -1)


    # test_images, validation_images, test_labels, validation_labels = train_test_split(test_images, test_labels, test_size=0.30, random_state=FLAGS.id)
    data_dim = train_images.shape[1:]

    if FLAGS.augmentation != 'none':

        total_synthetics = len(FLAGS.synthetic_paths)*FLAGS.synthetic_size
        # Generate Augmented and Synthetic Datasets
        synthetic_labels = list()
        for i in range(len(FLAGS.synthetic_paths)):
            synthetic_labels.append([i] * FLAGS.synthetic_size)
        synthetic_labels = np.array(synthetic_labels)
        synthetic_labels = np.reshape(synthetic_labels, newshape = total_synthetics)

        if classes == 4:
            synthetic_labels = tf.keras.utils.to_categorical(synthetic_labels)


        synthetic_data = list()
        for i in range(len(FLAGS.synthetic_paths)):
            synth_data = np.load(os.path.join(FLAGS.synthetic_paths[i], f'synthetic_data_{FLAGS.synthetic_size}.npz'))
            synth_data = synth_data.f.arr_0
            synthetic_data.append(synth_data)
        synthetic_data = np.array(synthetic_data)
        synthetic_data = np.reshape(synthetic_data,(total_synthetics,*data_dim))

        # synthetic_data = preprocess_binary_images(synthetic_data, normalize = normalize, binarization = binarization)

        augmented_data = np.concatenate((train_images,synthetic_data), axis = 0)
        augmented_data_labels = np.concatenate((train_labels, synthetic_labels))

        
        augmented_data = tf.expand_dims(augmented_data, axis = -1)

        


        # do checks for consistency among synthetic and real data
        assert synthetic_data.shape[1:] == train_images.shape[1:]
        assert np.max(synthetic_data[0]) <= 1.0
        assert np.max(train_images[0]) <= 1.0
        assert np.min(synthetic_data[0]) >= 0.
        assert np.min(train_images[0]) >= 0.




    METRICS = [
      #tf.keras.metrics.TruePositives(name='tp'),
      #tf.keras.metrics.FalsePositives(name='fp'),
      #tf.keras.metrics.TrueNegatives(name='tn'),
      #tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'),
      #tfma.metrics.BalancedAccuracy(name = 'bAC'),
    ]

    if FLAGS.model == 'AlexNet':

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(data_dim)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation = 'softmax')
        ])




    if FLAGS.model == 'cnn':


        model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = train_images.shape[1::]),
                #tf.keras.applications.DenseNet121(include_top=False),
                #dense_net,
                #tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512,activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(256,activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
            ])

        print('CNN Model')

    if FLAGS.model == 'mlp':

        if classes == 4:
            model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape = data_dim),
                    tf.keras.layers.Flatten(),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(512,activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    #tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(256,activation='relu'),
                    #tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(4, activation = 'softmax')
                ])

        if classes == 2:
            model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape = data_dim),
                    tf.keras.layers.Flatten(),
                    #tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(512,activation='relu'),
                    #tf.keras.layers.BatchNormalization(),
                    #tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(256,activation='relu'),
                    #tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                ])

        print('MLP Model')


    if FLAGS.model == 'logistic_regression':


        model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = data_dim),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
            ])

        print('MLP Model')

    # model = tf.keras.Sequential([
    #   tf.keras.layers.InputLayer(input_shape = data_dim),
    #   tf.keras.layers.Conv2D(filters = 8, kernel_size = (3,3),strides = 1),
    #   tf.keras.layers.BatchNormalization(),
    #   tf.keras.layers.LeakyReLU(),
    #   tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2),
    #   tf.keras.layers.Conv2D(filters = 16, kernel_size = (3,3),strides = 1),
    #   tf.keras.layers.BatchNormalization(),
    #   tf.keras.layers.LeakyReLU(),
    #   tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2),
    #   tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3),strides = 2),
    #   tf.keras.layers.BatchNormalization(),
    #   tf.keras.layers.LeakyReLU(),
    #   tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2),
    #   tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),strides = 2),
    #   tf.keras.layers.BatchNormalization(),
    #   tf.keras.layers.LeakyReLU(),
    #   tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = 2),
    #   tf.keras.layers.Flatten(),
    #   tf.keras.layers.Dense(256,activation='relu'),
    #   tf.keras.layers.Dense(100,activation='relu'),
    #   tf.keras.layers.Dense(1, activation = 'sigmoid')
    #   ])




    if classes == 2:
        model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS,
                )

    if classes == 4:
        model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
          loss = tf.keras.losses.CategoricalCrossentropy(),
          metrics=METRICS,
            )

    model.summary()

    def scheduler(epoch, lr):
        if epoch < 300:
            return lr
        else:
            return lr * tf.math.exp(-0.1)




    LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
    # checkpoint_filepath = os.path.join(path,'model_weights')

    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
        )

    if FLAGS.augmentation == 'none':

        history = model.fit(
            train_images,
            train_labels,
            batch_size=100,
            epochs=FLAGS.epochs,
            validation_data=(test_images, test_labels),
            callbacks = [early_stopping_monitor],
            verbose = 2 
            )

    if FLAGS.augmentation == 'synthetic_raw_mix':

        history = model.fit(
            augmented_data,
            augmented_data_labels,
            batch_size=100,
            epochs=FLAGS.epochs,
            validation_data=(test_images, test_labels),
            callbacks = [early_stopping_monitor],
            verbose = 2 
            )

    if FLAGS.augmentation == 'synthetic':

        history = model.fit(
            synthetic_data,
            synthetic_labels,
            batch_size=100,
            epochs=FLAGS.epochs,
            validation_data=(test_images, test_labels),
            callbacks = [early_stopping_monitor],
            verbose = 2 
            )

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_metrics(history):

        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])

            plt.legend()


    #Evaluating the model on the data
    plot_metrics(history)
    plt.savefig(os.path.join(path,'Metrics'))
    #plt.show()
    plt.clf()

    train_scores = model.evaluate(train_images, train_labels)
    # val_scores = model.evaluate(validation_images, validation_labels)
    test_scores = model.evaluate(test_images, test_labels)

    print("Training Accuracy: %.2f%%"%(train_scores[1] * 100))
    # print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100))
    print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))

    pred_labels = model.predict(test_images)
    #Print the classification report of the tested data

    #Since the labels are softmax arrays, we need to roundoff to have it in the form of 0s and 1s,
    #similar to the test_labels
    def roundoff(arr):
        """To round off according to the argmax of each predicted label array. """
        arr[np.argwhere(arr != arr.max())] = 0
        arr[np.argwhere(arr == arr.max())] = 1
        return arr

    for labels in pred_labels:
        labels = roundoff(labels)
 

    if classes == 4:
        pred_ls = np.argmax(pred_labels, axis=1)
        test_ls = np.argmax(test_labels, axis=1)

    if classes == 2:
        pred_ls = pred_labels
        # pred_ls = np.squeeze(pred_ls)
        test_ls = test_labels



    print(classification_report(test_labels, pred_labels, target_names=CLASSES))

    if classes == 2:
        train_predictions_baseline = model.predict(train_images, batch_size=32)
        test_predictions_baseline = model.predict(test_images, batch_size=32)

        def plot_cm(labels, predictions, p=0.5):
          cm = confusion_matrix(labels, predictions > p)
          plt.figure(figsize=(5,5))
          sns.heatmap(cm, annot=True, fmt="d")
          plt.title('Confusion matrix @{:.2f}'.format(p))
          plt.ylabel('Actual label')
          plt.xlabel('Predicted label')
          plt.savefig(os.path.join(path,'confusion_matrix'))


          sensitivity =  cm[1][1] / (cm[1][1] +  cm[1][0])
          specificity = cm[0][0] / (cm[0][0] + cm[0][1])

          balanced_accuracy = (sensitivity + specificity) * 0.5

          print('True Negatives: ', cm[0][0])
          print('False Positives: ', cm[0][1])
          print('False Negatives: ', cm[1][0])
          print('True Positives: ', cm[1][1])
          print('Balanced Accuracy', balanced_accuracy)
          print('sensitivity', sensitivity)
          print('specificity', specificity)
          return balanced_accuracy, sensitivity, specificity


        balanced_accuracy, sensitivity, specificity = plot_cm(test_labels, test_predictions_baseline)


        baseline_results = model.evaluate(test_images, test_labels,verbose=0)
        results = []

        for name, value in zip(model.metrics_names, baseline_results):
          print(name, ': ', value)
          results.append([name, ': ', value])
        results.append(['Balanced Accuracy', ': ', balanced_accuracy])
        results.append(['Sensitivity', ': ', sensitivity])
        results.append(['Specificity', ': ', specificity])
        results.append(['Training Accuracy', ': ', (train_scores[1] * 100)])
        results.append(['Testing Accuracy', ': ', (test_scores[1] * 100)])
        print()
        pd.DataFrame(results).to_csv(os.path.join(path,'results'),index=False)

    if classes == 4:
        conf_arr = confusion_matrix(test_ls, pred_ls)

        plt.figure(figsize=(13, 13), dpi=80, facecolor='w', edgecolor='k')

        ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)

        plt.title('Alzheimer\'s Disease Diagnosis')
        plt.xlabel('Prediction')
        plt.ylabel('Truth')
        plt.savefig(os.path.join(path,'confusion_matrix'))
        plt.clf()
        
        #Printing some other classification metrics
        print("Balanced Accuracy Score: {} %".format(round(BAS(test_ls, pred_ls) * 100, 2)))
        print("Matthew's Correlation Coefficient: {} %".format(round(MCC(test_ls, pred_ls) * 100, 2)))

        baseline_results = model.evaluate(test_images, test_labels,verbose=0)
        results = []

        for name, value in zip(model.metrics_names, baseline_results):
          print(name, ': ', value)
          results.append([name, ': ', value])
        results.append(['Balanced Accuracy', ': ', round(BAS(test_ls, pred_ls) * 100, 2)])
        results.append(["Matthew's Correlation Coefficient", ': ', round(MCC(test_ls, pred_ls) * 100, 2)])
        results.append(['Training Accuracy', ': ', (train_scores[1] * 100)])
        # results.append(['Validation Accuracy', ': ', (val_scores[1] * 100)])
        results.append(['Testing Accuracy', ': ', (test_scores[1] * 100)])
        print()

        pd.DataFrame(results).to_csv(os.path.join(path,'results'),index=False)


if __name__ == '__main__':
 app.run(main)









