import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp

from model import execute_action, generate_random_states, set_state, calculate_state, operating_cost

from uncertainties import *

# Dimensões do estado do sistema
# action_space_dim = 14
# dimensions = 27

# Define PPO2 neural network model
def build_ppo2_model(dimensions, action_space_dim):
    model = keras.Sequential([
        # layers.Input(shape=(dimensions,)),  # state is a n-dimensional array
        layers.Reshape((27, 1), input_shape=(dimensions,)),  # Reshape to 2D

        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_space_dim * 2),  # Output means and log_std for each action
    ])
    return model


def convert(lista):
    if isinstance(lista, list):
        np_list = np.array(lista, dtype=np.float32)
        tensor = tf.convert_to_tensor(np_list)
    elif isinstance(lista, np.ndarray):
        tensor = tf.convert_to_tensor(lista)
    return tensor

def convert_back(state):
    if isinstance(state, tf.Tensor):
        np_list = state.numpy()
        list_back = np_list.tolist()
    elif isinstance(state, np.ndarray):
        list_back = state.tolist()

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

def apply_policy(state, policy_model):

    # if not isinstance(state, tf.Tensor):
    #     state = convert(state)
    # print("Input Data Shape: Entry State", state.shape)

    # Apply policy
    policy_logits = policy_model(np.expand_dims(state, axis=0))[0]
    #print("Shape of logits:", policy_logits.shape)

    policy_logits_reshaped = tf.reshape(policy_logits,(27, 14, 2))

# Split logits into means and stds
    action_means, action_stds = tf.split(policy_logits_reshaped, 2, axis=1)

# Assuming you want to sample actions
    actions = tfp.distributions.Normal(loc=action_means, scale=action_stds).sample()
    #print("Shape of actions:", actions.shape)

    return actions

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

def ppo_policy_update(policy_model, states, actions, advantages):

    print("Shape of states:", states.shape)

    print("Shape of actions:", actions.shape)

    with tf.GradientTape() as tape:
        # Forward pass to calculate logits and policy distribution
        # policy_logits = policy_model(states)
        # policy_distribution = tfp.distributions.Categorical(logits=policy_logits) #arent those actions?

        # # actions = tf.reshape(actions, (-1,))
        
        # print("Shape of actions:", actions.shape)

        # # Calculate log probability of the selected actions under the current policy
        # log_prob_current_policy = policy_distribution.log_prob(actions)
        new_actions, surrogate_objectives =[],[]
        for x in range(len(states)):
            state, action, adv = states[x], actions[x], advantages[x]
            # print(type(state), type(action), type(adv))

            # act = apply_policy(state, policy_model)
            act = policy_model(np.expand_dims(state, axis=0))[0]
            print("Shape of cool actions:", action.shape, act.shape)


            new_actions.append(act)

            # Calculate the ratio of probabilities (new_policy / current_policy)
            # ratio = tf.exp(log_prob_current_policy)
            current_policy = tfp.distributions.Normal(loc=action[:, 0:1], scale=action[:, 1:])
            ratio = tf.exp(current_policy.log_prob(act) - current_policy.log_prob(action))

            # PPO Clipped Surrogate Objective
            clip_epsilon = 0.2
            surrogate_objective = tf.minimum(ratio * adv,
                                            tf.clip_by_value(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv)
            
            surrogate_objectives.append(surrogate_objective)

        # PPO Policy Optimization
        surrogate_objectives = tf.concat(surrogate_objectives, axis=0)
        policy_loss = -tf.reduce_mean(surrogate_objective)

        # Get gradients and update policy model parameters
        policy_gradients = tape.gradient(policy_loss, policy_model.trainable_variables)
        policy_optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        policy_optimizer.apply_gradients(zip(policy_gradients, policy_model.trainable_variables))

        return policy_model
    
# Example usage
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


def subset_samples(initial_state, sampled_states, policy_model, chain=None):

    states_list, actions_list, rewards_list, next_states_list= [], [], [], []
    # print(len(sampled_states), " :many states")
    for state in sampled_states:
        # Assume your environment provides a function to take actions and get the next state and reward
        #calculate_state(chain)
        # action = session.run(policy_model, feed_d ict={states: np.expand_dims(state, axis=0)})[0]
        state = convert(state)
        # print("Input Data Shape, sampled_state:", state.shape)

        action = policy_model(np.expand_dims(state, axis=0))[0]
        # print("Input Data Shape, generated action:", action.shape)

        action = convert_back(action)
        next_state, reward = take_action(action)

        action = convert(action)
        next_state = convert(next_state)
        # reward = convert(reward)

        # Append samples to the lists
        states_list.append(state)
        actions_list.append(action)
        rewards_list.append(reward)
        next_states_list.append(next_state)

        if chain:
            reset_state(chain, initial_state)

    # Convert lists to NumPy arrays
    sampled_states = np.array(states_list)
    actions = np.array(actions_list)
    rewards = np.array(rewards_list)
    next_states = np.array(next_states_list)

    return sampled_states, actions, rewards, next_states


def training(policy_model, value_model, initial_state, sampled_states, chain):

    values, next_values, advantages =[], [], []
    # Collect samples using the current policy
    states, actions, rewards, next_states = subset_samples(initial_state,sampled_states, policy_model, chain)

        # Calculate advantages and returns
    # print("Input Data Shape:", states.shape)

    for state in states:
        v = value_model(np.expand_dims(state, axis=0))[0]
        # v = apply_policy(state, value_model)
        values.append(v)
    for ns in next_states:
        nv = value_model(np.expand_dims(ns, axis=0))[0]
        # nv = apply_policy(ns, value_model)
        next_values.append(nv)

    # print("lens: ", len(values),len(next_values), len(rewards))
    for x in range(len(rewards)):    
        v, nv, r = values[x], next_values[x], rewards[x]
        # deltas = rewards + gamma * next_values * (1 - dones) - values
        adv = calculate_gae_advantages(v, nv, r, gamma, lambda_)
        advantages.append(adv)

    n_adv = []
    for adv in advantages:
    # Normalize advantages
        adv = (adv - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        n_adv.append(adv)

    try:

        updated_policy = ppo_policy_update(policy_model, states, actions, advantages)
        updated_value = ppo_policy_update(value_model, states, actions, advantages)        

        # # Update policy and value networks
        for _ in range(5):  # Number of optimization steps
            updated_policy = ppo_policy_update(updated_policy, states, actions, advantages)
            updated_value = ppo_policy_update(updated_value, states, actions, advantages)  
        
        return updated_policy, updated_value
    
    except:
        return policy_model, value_model
    
def best_action(policy_model, states):
        
    # policy_logits = policy_model(states)
    # policy_distribution = tfp.distributions.Categorical(logits=policy_logits)
    # sampled_actions = policy_distribution.sample()
    sampled_actions = apply_policy(states, policy_model)
    action_means = sampled_actions[:, :, 0]
    best_action_index = tf.argmax(action_means, axis=1)
    action_values = tf.linspace(0.0, 1.0, 14)
    # best_state = tf.gather(np.expand_dims(states, axis=0), best_action_index)
    best_action_value = tf.gather(action_values, best_action_index)
    best_action = best_action_value.numpy()
    return best_action


def take_action(action, chain=None, demand=None, timestep=None):
    if demand == None:
        demand = 400
    if timestep == None:
        timestep =1

    if chain == None:
        new_state = np.random.rand(27)
        reward = np.random.randint(-100, 101)

    else:
        updated_chain = execute_action(action, demand, timestep, chain)
        new_state = calculate_state(updated_chain)
        reward = operating_cost(updated_chain)

    return new_state, reward

def reset_state(chain, state):
    try:
        state = convert_back(state)
    except:
        pass
    return set_state(chain, state)

def standardize_data(state, mean=None, std=None):
    
    if mean is None:
        mean = np.mean(state, axis=0)
    if std is None:
        std = np.std(state, axis=0)

    standardized_data = (state - mean) / std
    return standardized_data

def execute_policy(policy_model, value_model, chain, initial_state, normalized_state, n):
    #generate samples
    sample_states = generate_random_states(n)
    # print(sample_states)
    # sample_states = convert(sample_states)
    # print(len(sample_states), type(sample_states[0]))

    #treina o modelo
    print("update")
    updated_policy, updated_value = training(policy_model, value_model, initial_state, sample_states, chain)

    try:
        selected_best_action = best_action(updated_policy, normalized_state)
    except:
        print(' unexpected dimension crash')
        new_state = np.random.uniform(0, 1, 27)
        selected_best_action = best_action(updated_policy, new_state)


    selected_best_action = convert_back(selected_best_action)
    print('found the best')

    return selected_best_action, updated_policy, updated_value


#test
def main():
    print('wobeli')

    policy_model = build_ppo2_model(27, 14)
    value_model = build_ppo2_model(27, 14)

    entry_state = np.random.rand(27)
    entry_state = standardize_data(entry_state) #[-1,1]
    # print(entry_state)
    # action = apply_policy(entry_state, policy_model)
    # print(action)
    # return

    print("Input Data Shape: Entry State", entry_state.shape)
    chain = None
    samples = [standardize_data(np.random.rand(27)) for _ in range(10)]
    print("Input Data Shape: Sample of many", samples[0].shape)
    print(samples[0])
    
    # print(type(policy_model))
    # return
    # s,a,r,n = subset_samples(entry_state, samples, policy_model, chain=None)
    # p,v = training(policy_model, value_model, entry_state, samples, chain)
    a = best_action(policy_model,entry_state)
    print(a)

    #testing frame

if __name__ == "__main__":
    main()

