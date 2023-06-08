import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial
import itertools
import math


from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, multiply, LeakyReLU, BatchNormalization, LayerNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Huber

from tqdm import tqdm

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import logging
tf.get_logger().setLevel(logging.ERROR)

class RandomWeightedAverage(tf.keras.layers.Layer):
    """
    A custom Keras layer to generate random weighted averages of input samples.

    Parameters:
    - batch_size: The batch size of the input samples.

    This class inherits from the `tf.keras.layers.Layer` base class and implements the necessary methods to generate random weighted averages of input samples. It is commonly used in generative adversarial networks (GANs) during the training process.

    Example usage:
    layer = RandomWeightedAverage(batch_size)
    output = layer(inputs)
    """

    def __init__(self, batch_size: int):
        """
        Initializes a new instance of the RandomWeightedAverage class.

        Parameters:
        - batch_size: The batch size of the input samples.
        """
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        """
        Generates random weighted averages of input samples.

        Parameters:
        - inputs: A list of input samples.

        Returns:
        - weighted_averages: Randomly weighted averages of the input samples.

        This method generates random weights using uniform distribution between 0 and 1. It then applies these weights to the input samples to calculate the weighted averages. The formula used is (alpha * input1) + ((1 - alpha) * input2), where alpha is the randomly generated weight.

        Note: This method assumes the use of TensorFlow and imports `tf`.

        Example usage:
        weighted_averages = layer.call(inputs)
        """
        alpha = tf.random.uniform((self.batch_size, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Parameters:
        - input_shape: The shape of the input samples.

        Returns:
        - output_shape: The shape of the output.

        This method computes and returns the output shape of the layer, which is the same as the shape of the first input sample.

        Example usage:
        output_shape = layer.compute_output_shape(input_shape)
        """
        return input_shape[0]



class WCGANGP():
    """
    Wasserstein Conditional Generative Adversarial Network with Gradient Penalty (WCGAN-GP) implementation.

    Parameters:
    - x_train: The input training data.
    - y_train: The target labels for the training data.
    - latent_dim: The dimensionality of the latent noise vector.
    - batch_size: The number of samples in each batch.
    - n_critic: The number of critic (discriminator) updates per generator update.
    - pre_trained: Boolean indicating whether the model is pre-trained or not.
    - pre_trained_epoch: The epoch at which pre-training was performed.
    - pre_trained_iteration: The iteration at which pre-training was performed.
    - pre_trained_on_training: Boolean indicating whether pre-training was performed on the training set.
    - pre_trained_on_training_rate: The rate at which pre-training was performed on the training set.
    - on_training_epoch: The epoch number for training.

    This class implements the Wasserstein Conditional Generative Adversarial Network with Gradient Penalty (WCGAN-GP). It includes methods to build the generator and critic models, compute the gradient penalty loss, compile and train the models.

    The class initializes the WCGAN-GP model with the provided parameters. It also sets up the generator and critic models, defines the optimizer, and compiles the models with the appropriate loss functions.

    Example usage:
    wcgan = WCGANGP(x_train, y_train, latent_dim, batch_size, n_critic)
    wcgan.train(epochs=10000)
    """

    def __init__(self, 
                 x_train, 
                 y_train, 
                 latent_dim, 
                 batch_size,
                 n_critic,
                 pre_trained = False,
                 pre_trained_epoch = None,
                 pre_trained_iteration = None,
                 pre_trained_on_training = False,
                 pre_trained_on_training_rate = None,
                 on_training_epoch = None,):
        
        self.x_train = x_train
        self.y_train = y_train
        self.original_x_train = x_train
        self.original_y_train = y_train
        self.pre_trained = pre_trained
        self.pre_trained_epoch = pre_trained_epoch
        self.pre_trained_iteration = pre_trained_iteration
        self.pre_trained_on_training = pre_trained_on_training
        self.pre_trained_on_training_rate = pre_trained_on_training_rate
        self.on_training_epoch = on_training_epoch
        
        self.num_classes = len(np.unique(y_train))
        self.data_dim = x_train.shape[1]
        
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        
        self.n_critic = n_critic
        
        self.history = {'critic_loss': [],
                        'generator_loss': [],}

        # Log training progress.
        self.losslog = []

        # Adam optimizer, suggested by original paper.
        optimizer = Adam(learning_rate=0.0005, beta_1=0.05, beta_2=0.9)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Freeze generator's layers while training critic.
        self.generator.trainable = False

        # Data input (real sample).
        real_data = Input(shape=self.data_dim)
        # Noise input (z).
        noise = Input(shape=(self.latent_dim,))
        # Label input.
        label = Input(shape=(1,))
        
        # Generate data based of noise (fake sample)
        fake_data = self.generator([noise, label])
        
        # Critic (discriminator) determines validity of the real and fake data.
        fake = self.critic([fake_data, label])
        valid = self.critic([real_data, label])
        
        # Construct weighted average between real and fake data.
        interpolated_data = RandomWeightedAverage(self.batch_size)([real_data, 
                                                                    fake_data])
        
        # Determine validity of weighted sample.
        validity_interpolated = self.critic([interpolated_data, label])
        
        
        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument.
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_data)
        # Keras requires function names.
        partial_gp_loss.__name__ = 'gradient_penalty' 
        
        self.critic_model = Model(inputs=[real_data, label, noise],
                            outputs=[valid, fake, validity_interpolated])
        
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])
        if self.pre_trained:
            self.pre_training()
            self.generator.trainable = False

        # For the generator we freeze the critic's layers.
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator.
        noise = Input(shape=(self.latent_dim,))
        
        # Add label to input.
        label = Input(shape=(1,))
        
        # Generate data based of noise.
        fake_data = self.generator([noise, label])

        # Discriminator determines validity.
        valid = self.critic([fake_data, label])

        self.optimizer = optimizer

        # Define generator model.
        self.generator_model = Model([noise, label], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, 
                                     optimizer=self.optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes the gradient penalty loss for a given batch of samples.

        Parameters:
        - y_true: The true labels/targets for the samples.
        - y_pred: The predicted labels/targets for the samples.
        - averaged_samples: The averaged samples used for gradient penalty calculation.

        Returns:
        - gradient_penalty: The computed gradient penalty loss.

        This function calculates the gradient penalty loss for a given batch of samples. It is commonly used in generative adversarial networks (GANs) to enforce smoothness in the learned model. The gradient penalty loss penalizes the model if the gradient of the discriminator with respect to the interpolated samples deviates from a target value.

        The function computes the gradients of the discriminator's predictions `y_pred` with respect to the `averaged_samples`. It then calculates the Euclidean norm of the gradients, which measures the magnitude of the gradients. The function applies the formula `lambda * (1 - ||grad||)^2` to each individual sample, where `lambda` is a hyperparameter and `||grad||` is the Euclidean norm of the gradients. Finally, the mean of the gradient penalties over the batch samples is returned as the loss.

        Note: This function assumes the use of Keras backend (e.g., TensorFlow) and imports `K` from it.

        Example usage:
        loss = gradient_penalty_loss(y_true, y_pred, averaged_samples)
        """

        gradients = K.gradients(y_pred, averaged_samples)[0]

        # Compute the euclidean norm by squaring...
        gradients_sqr = K.square(gradients)
        # ...summing over the rows...
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        # ...and taking the square root
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)

        # Compute lambda * (1 - ||grad||)^2 for each single sample
        gradient_penalty = K.square(1 - abs(gradient_l2_norm))

        # Return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)



    def wasserstein_loss(self, y_true, y_pred):
        """
        Computes the Wasserstein loss for a given batch of samples.

        Parameters:
        - y_true: The true labels/targets for the samples.
        - y_pred: The predicted labels/targets for the samples.

        Returns:
        - wasserstein_loss: The computed Wasserstein loss.

        This function calculates the Wasserstein loss for a given batch of samples. It is commonly used in Wasserstein GANs (WGANs) to train the generator and discriminator. The Wasserstein loss measures the distance between the true and predicted label distributions, providing a more stable and meaningful training signal compared to traditional GAN losses.

        The function computes the element-wise product of `y_true` and `y_pred`, and then takes the mean of the resulting tensor. This represents the expectation over the joint distribution of the true and predicted labels. The resulting value is returned as the Wasserstein loss.

        Note: This function assumes the use of Keras backend (e.g., TensorFlow) and imports `K` from it.

        Example usage:
        loss = wasserstein_loss(y_true, y_pred)
        """

        return K.mean(y_true * y_pred)


    def build_generator(self):
        """
        Build and return the generator model.

        Returns:
        - generator_model: The constructed generator model.

        This function builds a generator model, which is responsible for generating synthetic data samples. The generator model typically takes random noise as input and outputs generated data samples.

        The generator model is constructed using a Sequential model from Keras. It consists of multiple dense (fully connected) layers with SELU activation functions, batch normalization, and dropout regularization. The output layer uses the sigmoid activation function to ensure the generated data samples are between 0 and 1.

        The noise and label inputs are defined using Keras Input layers. The label input is embedded into one-hot encoded vectors using an Embedding layer. The noise and embedded labels are then multiplied element-wise to create the model input. The generator model is then defined to take this model input and produce the generated data samples.

        Example usage:
        generator = build_generator()
        """

        model = Sequential(name="Generator")
        
        # First hidden layer.
        model.add(Dense(128, input_dim=self.latent_dim, activation='selu'))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Second hidden layer.
        model.add(Dense(256, activation='selu'))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Third hidden layer.
        model.add(Dense(512, activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Output layer.
        model.add(Dense(self.data_dim, activation="sigmoid"))
        
        model.summary() 
        
        # Noise and label input layers.
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype="int32")
        
        # Embed labels into onehot encoded vectors.
        label_embedding = Flatten()(Embedding(self.num_classes, 
                                              self.latent_dim)(label))
        
        # Multiply noise and embedded labels to be used as model input.
        model_input = multiply([noise, label_embedding])
        
        generated_data = model(model_input)

        return Model([noise, label], generated_data, name="Generator")

    def build_critic(self):
        """
        Builds and returns the Critic model.

        Returns:
        - model: The built Critic model.

        This method constructs the Critic model, which is commonly used in adversarial learning setups such as Generative Adversarial Networks (GANs). The Critic model evaluates the generated samples and provides a measure of their quality.

        The Critic model is a feedforward neural network with several hidden layers. Each hidden layer consists of a dense layer with linear activation followed by a leaky rectified linear unit (LeakyReLU) activation function with a fixed negative slope (alpha=0.2). The output layer has a single node with linear activation.

        The model architecture is as follows:
        - Input layer: Accepts samples with dimension `self.data_dim`.
        - Hidden layer 1: Dense layer with 512 units and linear activation, followed by LeakyReLU activation.
        - Hidden layer 2: Dense layer with 256 units and linear activation, followed by LeakyReLU activation.
        - Hidden layer 3: Dense layer with 128 units and linear activation, followed by LeakyReLU activation.
        - Output layer: Dense layer with 1 unit and linear activation.

        The model is then summarized, displaying the layer information. Additionally, two inputs are defined:
        - `generated_sample`: Artificial data input of shape `self.data_dim`.
        - `label`: Label input of shape (1,) with data type "int32".

        The `label` input is embedded as a one-hot vector using an Embedding layer, which maps each label to a vector representation. The label embedding is then flattened. The generated sample is multiplied element-wise with the label embedding to obtain the input for the Critic model.

        Finally, the Critic model is constructed using the defined inputs and the validity output is obtained by passing the model input through the constructed model. The resulting model is returned.

        Note: This method assumes the use of Keras and imports required classes and functions such as `Sequential`, `Dense`, `LeakyReLU`, `Embedding`, `Input`, `Flatten`, and `multiply`.

        Example usage:
        critic = build_critic()
        """

        model = Sequential(name="Critic")

        # First hidden layer.
        model.add(Dense(512, input_dim=self.data_dim, activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        
        # Second hidden layer.        
        model.add(Dense(256, activation='linear'))
        model.add(LeakyReLU(alpha=0.2))
        
        # Third hidden layer.
        model.add(Dense(128, activation='linear'))
        model.add(LeakyReLU(alpha=0.2))

        # Output layer with linear activation.
        model.add(Dense(1, activation='linear'))

        model.summary()
        
        # Artificial data input.
        generated_sample = Input(shape=self.data_dim)
        # Label input.
        label = Input(shape=(1,), dtype="int32")
        
        # Embedd label as onehot vector.
        label_embedding = Flatten()(Embedding(self.num_classes, 
                                              self.data_dim)(label))
        
        # Multiply fake data sample with label embedding to get critic input.
        model_input = multiply([generated_sample, label_embedding])
        
        validity = model(model_input)

        return Model([generated_sample, label], validity, name="Critic")

    def train(self, epochs):
        """
        Trains the WCGAN model for a specified number of epochs.

        Parameters:
        - epochs: The number of training epochs.

        This method trains the WCGAN model by iterating over the specified number of epochs. It performs the training in batches, with each batch consisting of real samples and generated samples.

        The method initializes the training set and sets the adversarial ground truths for the discriminator. It calculates the number of batches based on the training data size and batch size. It also handles any overhead data that does not fit into complete batches.

        Within each epoch, the method resets the training set and selects random overhead rows that do not fit into batches. It removes these random overhead rows from the training set. The training data is then split into batches for processing.

        For each batch, the method performs training on the critic model by training the discriminator using real samples, generated samples, and noise. It repeats the training process for a specified number of critic steps (`n_critic`).

        After training the critic, the method generates a sample of artificial labels and trains the generator model using the generated noise and labels.

        The losses for both the discriminator and generator are recorded in `self.losslog`. The current epoch's critic loss (`d_loss`) and generator loss (`g_loss`) are also printed for monitoring.

        The method updates the `self.history` dictionary with the critic loss and generator loss for each epoch. If the WCGAN model was pre-trained on training data, the method performs pre-training based on the specified rate.

        Example usage:
        wcgan = WCGAN()
        wcgan.train(100)
        """

        self.x_train = self.original_x_train.copy()
        self.y_train = self.original_y_train.copy()

        # Adversarial ground truths.
        valid = -(np.ones((self.batch_size, 1)))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))
        
        # Number of batches.
        self.n_batches = math.floor(self.x_train.shape[0] / self.batch_size)

        overhead = self.x_train.shape[0] % self.batch_size
        
        for epoch in range(epochs):
                
            # Reset training set.
            self.x_train = self.original_x_train.copy()
            self.y_train = self.original_y_train.copy()

            # Select random overhead rows that do not fit into batches.
            rand_overhead_idx = np.random.choice(range(self.x_train.shape[0]), 
                                                 overhead, 
                                                 replace=False)

            # Remove random overhead rows.
            self.x_train = np.delete(self.x_train, rand_overhead_idx, axis=0)
            self.y_train = np.delete(self.y_train, rand_overhead_idx, axis=0)

            # Split training data into batches.
            x_batches = np.split(self.x_train, self.n_batches)
            y_batches = np.split(self.y_train, self.n_batches)
            
            for x_batch, y_batch, i in tqdm(zip(x_batches, 
                                                y_batches, 
                                                range(self.n_batches))):
        
                for _ in range(self.n_critic):

                    # Generate random noise.
                    noise = np.random.normal(0, 1, (self.batch_size, 
                                                    self.latent_dim))
                             
                    # Train the critic.
                    d_loss = self.critic_model.train_on_batch(
                        [x_batch, y_batch, noise],                                      
                        [valid, fake, dummy])


                # Generate sample of artificial labels.
                generated_labels = np.random.randint(0, self.num_classes, self.batch_size).reshape(-1, 1)

                # Train generator.
                g_loss = self.generator_model.train_on_batch([noise, generated_labels], 
                                                             valid)

                self.losslog.append([d_loss[0], g_loss])
                
                DLOSS = "%.4f" % d_loss[0]
                GLOSS = "%.4f" % g_loss
                
                # if i % 100 == 0:
            print(f"Epoch: {epoch+1}/{epochs} critic_loss: {DLOSS} generator_loss: {GLOSS}")
            self.history['critic_loss'].append(d_loss[0])
            self.history['generator_loss'].append(g_loss)
            if self.pre_trained_on_training:
                if epoch % int(epochs * self.pre_trained_on_training_rate) == 0:
                    self.pre_training(on_training=True)


    def plot_history(self):
        """
        Plots the training history of a WCGAN model.

        This method plots the generator and critic loss values recorded during the training process. It provides insights into the model's learning progress and the convergence of the generator and critic.

        Example usage:
        wcgan.plot_history()
        """

        len_data = [i for i in range(len(self.history['generator_loss']))]
        plt.plot(len_data, self.history['generator_loss'], color='blue', label='generator_loss')
        plt.plot(len_data, self.history['critic_loss'], color='red', label='critic_loss')
        plt.title('WCGAN Loss')
        plt.legend()
        plt.show()


    def pre_training(self, on_training=False):
        """
        Pre-trains the Generator and Critic models.

        Parameters:
        - on_training (bool): Flag indicating if pre-training is performed during the main training phase. Default is False.

        This method performs pre-training of the Generator and Critic models. It includes multiple iterations of training the Generator and Critic models alternatively.

        During each iteration, the Generator is trained with the Huber loss and Adam optimizer. Random noise data is generated, and the Generator is fit to map this noise data and original labels to the original input data. The fit is performed for a fraction of the epoch count.

        After training the Generator, a subset of the original data is sampled by removing a random overhead of samples. The remaining data is used to train the Critic model. The Critic model is fit using the Wasserstein loss and three sets of labels: valid, fake, and dummy. The fit is performed for another fraction of the epoch count.

        At the end of each iteration, the Generator and Critic losses are printed.

        If `on_training` is True, the pre-training is performed as part of the main training phase. Otherwise, the pre-training iteration and epoch counts are used.

        Note: This method assumes the use of certain attributes and parameters that are not explicitly defined within the code provided.

        Example usage:
        pre_training()  # Perform pre-training during the main training phase
        pre_training(on_training=True)  # Perform pre-training separately from the main training phase
        """

        def wasserstein_loss(y_true, y_pred):
            return np.mean(y_true) * np.mean(y_pred)
        
        print('Pre-trained Generator & Critic Model')
        if on_training:
            iteration = 1
            epoch = self.on_training_epoch
        else:
            iteration = self.pre_trained_iteration
            epoch = self.pre_trained_epoch
        for i in range(iteration):
            self.generator.trainable = True
            self.generator.compile(loss=Huber(), optimizer='adam')
            noise_data = np.random.normal(0, 1, (self.original_x_train.shape[0], 
                                                 self.latent_dim))
            self.generator.fit([noise_data, self.original_y_train], 
                               self.original_x_train, 
                               batch_size=self.original_x_train.shape[0], 
                               epochs=epoch//4, 
                               verbose=0)
            history = self.generator.fit([noise_data, self.original_y_train], 
                                         self.original_x_train, 
                                         batch_size=self.batch_size, 
                                         epochs=epoch, 
                                         validation_split=0.2, 
                                         verbose=0)
            gen_loss = history.history['loss'][-1]
            overhead = self.original_x_train.shape[0] % self.batch_size
            rand_overhead_idx = np.random.choice(range(self.original_x_train.shape[0]), 
                                                 overhead, 
                                                 replace=False)
            x_train = np.delete(self.original_x_train, rand_overhead_idx, axis=0)
            y_train = np.delete(self.original_y_train, rand_overhead_idx, axis=0)
            valid = -(np.ones((x_train.shape[0], 1)))
            fake =  np.ones((x_train.shape[0], 1))
            dummy = np.zeros((x_train.shape[0], 1))
            noise_data = np.random.normal(0, 1, (x_train.shape[0], self.latent_dim))
            self.generator.trainable = False
            history = self.critic_model.fit([x_train, y_train, noise_data], 
                                            [valid, fake, dummy], 
                                            batch_size=self.batch_size, 
                                            epochs=epoch//10, 
                                            verbose=0)
            critic_loss = history.history['loss'][-1]
            print(f'Pretrained Iteration {i+1}/{self.pre_trained_iteration} Generator Loss: {gen_loss} Critic_Loss: {critic_loss}')



        

    def generate_data(self, n: int):
        """
        Generates synthetic data samples using the WCGAN generator.

        Parameters:
        - n: The number of synthetic samples to generate.

        Returns:
        - generated_data: The generated synthetic data samples.
        - generated_labels: The corresponding labels for the generated data.

        This function generates synthetic data samples using the WCGAN (Wasserstein Conditional Generative Adversarial Network) generator. It takes an input parameter `n` specifying the number of synthetic samples to generate.

        The function starts by creating a copy of the original training labels `tmp_y_train`. It then calculates the distribution ratio of each label in the dataset, storing the results in the `label_ratios` dictionary. This ratio represents the proportion of each label in the original dataset.

        Next, the function generates random noise using a normal distribution with mean 0 and standard deviation 1. The shape of the noise is `(n, self.latent_dim)`.

        The function proceeds to create synthetic data samples by sampling labels based on the label ratios. It uses a list comprehension with `round(ratio*n)` to determine the number of samples for each label. The resulting list of sampled labels is flattened using `itertools.chain` and converted into a numpy array.

        Finally, the WCGAN generator is used to generate artificial data samples by passing the generated noise and sampled labels as inputs to the generator's `predict` method. The function returns the generated data samples (`generated_data`) and the corresponding labels (`generated_labels`) as a tuple.

        Note: This function assumes the existence of a WCGAN generator object and requires the import of necessary libraries (e.g., `numpy`, `itertools`).

        Example usage:
        generated_data, generated_labels = generate_data(1000)
        """

        tmp_y_train = self.original_y_train.copy()
        # Get distribution ratio of each label in the dataset.
        label_ratios = {label: len(
            tmp_y_train[tmp_y_train == label])/tmp_y_train.shape[0] for label in np.unique(tmp_y_train)}

        noise = np.random.normal(0, 1, (n, self.latent_dim))

        # Create synthetic data samples
        sampled_labels = [
            np.full(round(ratio*n), label).tolist()
            for label, ratio in label_ratios.items()
        ]

        # Convert list to numpy array.
        sampled_labels = np.array((list(itertools.chain(*sampled_labels))))

        # Use CGAN to generate aritficial data.
        return self.generator.predict([noise, sampled_labels]), sampled_labels.flatten()

    def save_model(self, prefix_name=None, path="./"):
        """
        Saves the generator and critic models to disk.

        Parameters:
        - prefix_name (str): Optional prefix name to prepend to the saved model files. Default is None.
        - path (str): Path to the directory where the models should be saved. Default is "./".

        This method saves the generator and critic models of a GAN to disk. The saved models can be later loaded and used for generating new samples or further training.

        The method first creates a directory to store the saved models, if it does not exist already. The directory is created at the specified `path` or "./" by default.

        Next, the generator and critic models are saved in the models directory. The saved model files have names in the format "{prefix_name}_generator.h5" and "{prefix_name}_critic.h5", where `prefix_name` is an optional prefix string provided as an argument to the method. If no prefix name is provided, the default names "generator.h5" and "critic.h5" are used.

        Example usage:
        gan.save_model(prefix_name="my_gan", path="./saved_models/")
        """

        models_dir = os.path.join(path, "models")
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        self.generator.save(os.path.join(models_dir, f"{prefix_name}_generator.h5"))
        self.critic.save(os.path.join(models_dir, f"{prefix_name}_critic.h5"))

    def load_model(self, prefix_name=None, path="./"):
        """
        Loads the generator and critic models from saved files.

        Parameters:
        - prefix_name: The prefix name used for the saved model files. Default is None.
        - path: The path to the directory where the models are saved. Default is "./".

        This method loads the generator and critic models from saved files. The saved models are assumed to be in the specified directory (`path`) with the provided prefix name (`prefix_name`) appended to the filenames.

        The method first constructs the directory path for the models based on the specified `path`. It then checks if the directory exists and raises a `ValueError` if it doesn't.

        Next, it loads the generator model by calling `tf.keras.models.load_model()` with the path to the generator model file. The file name is constructed by appending "_generator.h5" to the prefix name.

        Similarly, the method loads the critic model by calling `tf.keras.models.load_model()` with the path to the critic model file. The file name is constructed by appending "_critic.h5" to the prefix name.

        Note: This method assumes the use of TensorFlow and imports the `os` and `tf` modules.

        Example usage:
        my_model.load_model(prefix_name="my_model", path="./saved_models/")
        """

        models_dir = os.path.join(path, "models")

        if not os.path.exists(models_dir):
            raise ValueError(f"Models directory '{models_dir}' does not exist.")

        self.generator = tf.keras.models.load_model(os.path.join(models_dir, f"{prefix_name}_generator.h5"))
        self.critic = tf.keras.models.load_model(os.path.join(models_dir, f"{prefix_name}_critic.h5"))