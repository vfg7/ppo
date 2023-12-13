import random, math
import numpy as np
from scipy.stats import stats, mode
import matplotlib.pyplot as plt


#generate stochastic values for experimental scenarios

def stochastic_demand(min, max, z, t, total_timesteps, p):

    s = sinusoidal(min, max, z, total_timesteps,t)
    d = disturbance(0, p)
    print(s,p)

    return clip(s+d,max,min)

def clip(x, max, min):

    if x < min:
        return min
    elif x > max:
        return max
    else: 
        return x

def sinusoidal(min, max, z, total_timesteps, given_timestep=None):

    print(given_timestep)
    if given_timestep:
        return min + (max-min)/2 * (1+math.sin(2*z*given_timestep*math.pi/total_timesteps))
    
    else:
        stochastics = []
        for t in range(total_timesteps):
            s = min + (max-min)/2 * (1+math.sin(2*z*t*math.pi/total_timesteps))
            stochastics.append(s)
            #what do i return? and what about z?
        return stochastics


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
    min, max, z, t, total_timesteps, p = 100,400,6,60,360,60
    a = uniform_demand(min,max)
    print(a)
    a = stochastic_demand(min, max, z,t, total_timesteps, p)
    print(a)

if __name__ == "__main__":
    main()