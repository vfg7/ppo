import random, math
import numpy as np
from scipy.stats import stats, mode
import matplotlib.pyplot as plt


#generate stochastic values for experimental scenarios

def stochastic_demand(min, max, freq, t, total_timesteps, mean, std):

    s = sinusoidal(min, max, freq, total_timesteps,t)
    d = disturbance(mean, std)
    print(s,d)

    return clip(s+d,max,min)

def clip(x, max, min):

    if x < min:
        return min
    elif x > max:
        return max
    else: 
        return x

def sinusoidal(min, max, freq, total_timesteps, given_timestep=None):

    if given_timestep==0:
        given_timestep =1
    
    return min + (max-min)/2 * (1+math.sin(2*freq*given_timestep*math.pi/total_timesteps))


def disturbance(mean, std_dev):

    return np.random.normal(mean, std_dev)

def uniform_demand(lower, upper):

    return upper + np.random.uniform(lower, upper)

def normalize_minus(value, min_val, max_val):
    # Ensure min_val is not equal to max_val to avoid division by zero
    if min_val == max_val:
        raise ValueError("min_val and max_val must be different for normalization.")
    
    # Scale the value to the interval [-1, 1]
    normalized_value = 2 * (value - min_val) / (max_val - min_val) - 1
    
    return normalized_value

def normalize_to_zero_one(value, min_val, max_val):
    # Ensure min_val is not equal to max_val to avoid division by zero
    if min_val == max_val:
        raise ValueError("min_val and max_val must be different for normalization.")
    
    # Scale the value to the interval [0, 1]
    normalized_value = (value - min_val) / (max_val - min_val)
    
    return normalized_value

#test
def main():
    # print(disturbance(3,4))
    min, max, z, t, total_timesteps, p = 100,400,0.1,1,360,60
    # a = uniform_demand(min,max)
    # print(a)
    a = stochastic_demand(min, max, z,t, total_timesteps, 200, p)
    print(a)

if __name__ == "__main__":
    main()