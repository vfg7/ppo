import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

from uncertainties import *

# Dimensões do estado do sistema
action_space_dim = 14
dimensions = 27

# Define PPO2 neural network model
def build_ppo2_model(dimensions):
    model = keras.Sequential([
        layers.Input(shape=(dimensions,)),  # state is a n-dimensional array
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_space_dim * 2),  # Output means and log_std for each action
    ])
    return model

# hyperparameters
gamma = 0.99
lambda_ = 0.95
clip_ratio = 0.2
epochs = 10 #testar
mini_batch_size = 64

#taxas de aprendizado
policy_lr = 1e-3
value_lr = 1e-3

#loss parameters
c1 = 0.05
c2 = 0.01

# Build models
policy_model = build_ppo2_model()
value_model = build_ppo2_model()

# Optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_lr)
value_optimizer = keras.optimizers.Adam(learning_rate=value_lr)

# Placeholder tensors for inputs
states = tf.placeholder(tf.float32, shape=(None, dimensions))
actions = tf.placeholder(tf.float32, shape=(None, action_space_dim))
advantages = tf.placeholder(tf.float32, shape=(None,))
returns = tf.placeholder(tf.float32, shape=(None,))

#perdas

# PPO2 Loss functions
policy_logits = policy_model(states)
means, log_stds = tf.split(policy_logits, 2, axis=-1)
stds = tf.exp(log_stds)
policy_distribution = tfp.distributions.Normal(means, stds)

old_policy_logits = policy_model(states)
old_means, old_log_stds = tf.split(old_policy_logits, 2, axis=-1)
old_stds = tf.exp(old_log_stds)
old_policy_distribution = tfp.distributions.Normal(old_means, old_stds)

#  probabilities and log probabilities

action_probs = policy_distribution.prob(actions)
old_action_probs = old_policy_distribution.prob(actions)
action_log_probs = policy_distribution.log_prob(actions)
old_action_log_probs = old_policy_distribution.log_prob(actions)

#policy update

# PPO2 Surrogate Loss
ratio = tf.exp(action_log_probs - old_action_log_probs)
surrogate_loss_1 = ratio * advantages
surrogate_loss_2 = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
surrogate_loss = -tf.reduce_mean(tf.minimum(surrogate_loss_1, surrogate_loss_2))

# PPO2 Value Function Loss
values = value_model(states)
value_loss = tf.reduce_mean(tf.square(values - returns))

# PPO2 Entropy Loss (optional, can help with exploration)
entropy_loss = -tf.reduce_mean(policy_distribution.entropy())

# PPO2 Total Loss
total_loss = surrogate_loss + c1 * value_loss - c2 * entropy_loss


# PPO2 Optimization
policy_params = policy_model.trainable_variables
value_params = value_model.trainable_variables

policy_gradients = tf.gradients(total_loss, policy_params)
value_gradients = tf.gradients(value_loss, value_params)

policy_optimizer = tf.train.AdamOptimizer(learning_rate=policy_lr)
value_optimizer = tf.train.AdamOptimizer(learning_rate=value_lr)

update_policy = policy_optimizer.apply_gradients(zip(policy_gradients, policy_params))
update_value = value_optimizer.apply_gradients(zip(value_gradients, value_params))

# PPO2 Training Loop
# input stochastic samples
# demanda e leadtime são gerados estocasticamente e o modelo da cadeia calcula o estado

# Initialize TensorFlow session
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #nonono

for epoch in range(epochs):
    # Collect samples using the current policy
    states, actions, rewards, next_states, dones = subset_samples(policy_model)

    # Calculate advantages and returns
    values = value_model(states)
    next_values = value_model(next_states)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = calculate_gae_advantages(deltas, gamma, lambda_)

    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # Update policy and value networks
    for _ in range(5):  # Number of optimization steps
        sess.run(update_policy, feed_dict={states: states, actions: actions, advantages: advantages})
        sess.run(update_value, feed_dict={states: states, returns: rewards + gamma * next_values * (1 - dones)})

# Continue training or use the policy for inference
