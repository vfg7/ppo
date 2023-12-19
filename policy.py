import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp

from model import execute_action, generate_random_states, set_state, calculate_state, operating_cost

from uncertainties import *
from supply_chain import *

# Dimensões do estado do sistema
# action_space_dim = 14
# dimensions = 27

# Define PPO2 neural network model
def build_ppo2_model(dimensions, action_space_dim):
    model = keras.Sequential([
        layers.Input(shape=(dimensions,)),  # state is a n-dimensional array
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_space_dim * 2),  # Output means and log_std for each action
    ])
    return model


def convert(lista):
    # Convert lists to NumPy arrays
    np_list = np.array(lista, dtype=np.float32)
    # Convert NumPy arrays to TensorFlow tensors
    tensor = tf.convert_to_tensor(np_list)
    return tensor

def convert_back(tensor):

    np_list = tensor.numpy()
    list_back = np_list.tolist()
    return list_back



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
# policy_model = build_ppo2_model(d,a)
# value_model = build_ppo2_model(d,a)

def apply_policy(states, policy_model):

    # Apply policy
    policy_logits = policy_model(states)
    means, log_stds = tf.split(policy_logits, 2, axis=-1)
    stds = tf.exp(log_stds)
    policy_distribution = tfp.distributions.Normal(means, stds)

    # sampled_actions = policy_distribution.sample() #on run

    return policy_distribution

def generate_new_policy(policy_model, value_model, states, values, actions, advantages, returns):
    policy_logits = policy_model(states)
    means, log_stds = tf.split(policy_logits, 2, axis=-1)
    stds = tf.exp(log_stds)
    policy_distribution = tfp.distributions.Normal(means, stds)

    old_policy_logits = policy_model(states)
    old_means, old_log_stds = tf.split(old_policy_logits, 2, axis=-1)
    old_stds = tf.exp(old_log_stds)
    old_policy_distribution = tfp.distributions.Normal(old_means, old_stds)

    total_loss, value_loss = calculate_losses(policy_distribution, old_policy_distribution, values, actions, advantages, returns)

    upol, uval = optimize(policy_model,value_model, total_loss, value_loss)

    return upol, uval

def calculate_losses(current_policy_distribution, new_policy_distribution, values, actions, advantages, returns):
    #as tensors
    # action_probs = current_policy_distribution.prob(actions)
    # new_action_probs = new_policy_distribution.prob(actions)
    action_log_probs = current_policy_distribution.log_prob(actions)
    new_action_log_probs = new_policy_distribution.log_prob(actions)

    # PPO2 Surrogate Loss
    ratio = tf.exp(new_action_log_probs - action_log_probs)
    surrogate_loss_1 = ratio * advantages
    surrogate_loss_2 = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    surrogate_loss = -tf.reduce_mean(tf.minimum(surrogate_loss_1, surrogate_loss_2))

    # PPO2 Value Function Loss
    value_loss = tf.reduce_mean(tf.square(values - returns))

    # PPO2 Entropy Loss (optional, can help with exploration)
    entropy_loss = -tf.reduce_mean(new_policy_distribution.entropy())

    # PPO2 Total Loss
    total_loss = surrogate_loss + c1 * value_loss - c2 * entropy_loss

    return total_loss, value_loss

def optimize(policy_model,value_model, total_loss, value_loss):
    # PPO2 Optimization
    policy_params = policy_model.trainable_variables
    value_params = value_model.trainable_variables

    policy_gradients = tf.gradients(total_loss, policy_params)
    value_gradients = tf.gradients(value_loss, value_params)

    policy_optimizer = tf.train.AdamOptimizer(learning_rate=policy_lr)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=value_lr)

    update_policy = policy_optimizer.apply_gradients(zip(policy_gradients, policy_params))
    update_value = value_optimizer.apply_gradients(zip(value_gradients, value_params))

    return update_policy, update_value

# PPO2 Training Loop
# input stochastic samples
# demanda e leadtime são gerados estocasticamente e o modelo da cadeia calcula o estado

def calculate_gae_advantages(values, next_values, rewards, gamma, lambda_):
    
    # deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    deltas = rewards + gamma * next_values - values

    advantages = np.zeros_like(deltas, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(deltas))):
        running_add = running_add * gamma * lambda_ + deltas[t]
        advantages[t] = running_add
    
    return advantages


def subset_samples(initial_state, policy_model, session, num_samples, chain):

    states_list, actions_list, rewards_list, next_states_list= [], [], [], []

    for state in range(num_samples):
        # Assume your environment provides a function to take actions and get the next state and reward
        #calculate_state(chain)
        action = session.run(policy_model, feed_dict={states: np.expand_dims(state, axis=0)})[0]
        action = convert_back(action)
        next_state, reward = take_action(action)

        action = convert(action)
        next_state = convert(next_state)
        reward = convert(reward)

        # Append samples to the lists
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        next_states_list.append(next_state)
  
        reset_state(chain, initial_state)

    # Convert lists to NumPy arrays
    states = np.array(states_list)
    actions = np.array(actions_list)
    rewards = np.array(rewards_list)
    next_states = np.array(next_states_list)

    return states, actions, rewards, next_states


def training(policy_model, value_model, initial_state, states, chain):

    # Initialize TensorFlow session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 

    for _ in range(epochs):
        # Collect samples using the current policy
        states, actions, rewards, next_states = subset_samples(initial_state, policy_model, sess, len(states), chain)

        # Calculate advantages and returns
        values = sess.run(value_model, feed_dict={states: np.expand_dims(states, axis=0)})[0]
        next_values = sess.run(value_model, feed_dict={states: np.expand_dims(next_states, axis=0)})[0]
        # deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = calculate_gae_advantages(values, next_values, rewards, gamma, lambda_)

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        update_policy, update_value = generate_new_policy(policy_model, value_model, states, 
                                                          values, actions, advantages, rewards)

        # # Update policy and value networks
        for _ in range(5):  # Number of optimization steps
            sess.run(update_policy, feed_dict={states: states, actions: actions, advantages: advantages})
            sess.run(update_value, feed_dict={states: states, rewards: rewards + gamma * next_values})
        
        return update_policy, update_value


def best_state(policy_model, states):
        
    policy_logits = policy_model(states)
    policy_distribution = tfp.distributions.Categorical(logits=policy_logits)
    sampled_actions = policy_distribution.sample()

    best_state_index = tf.argmax(sampled_actions, axis=1)
    best_state = tf.gather(states, best_state_index)

    with tf.Session() as sess:

        best_state_value = sess.run(best_state, feed_dict={states})
    
    # print(best_state_value)
    return best_state_value


def take_action(action, demand, timestep, chain):
    updated_chain = execute_action(action, demand, timestep, chain)
    new_state = calculate_state(updated_chain)
    reward = operating_cost(updated_chain)

    return new_state, reward

def reset_state(chain, state):
    return set_state(chain, state)

def execute_policy(policy_model, value_model, chain, initial_state, n):
    #generate samples
    sample_states = generate_random_states(n)
    sample_states = convert(sample_states)
    #treina o modelo
    updated_policy, updated_value = training(policy_model, value_model, initial_state, sample_states, chain)

    best_action = best_state(updated_policy, initial_state)
    best_action = convert_back(best_action)

    return best_action, updated_policy, updated_value


#test
def main():

    return

if __name__ == "__main__":
    main()

