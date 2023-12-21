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


def normalize(value, min_val, max_val):
    # Ensure min_val is not equal to max_val to avoid division by zero
    if min_val == max_val:
        raise ValueError("min_val and max_val must be different for normalization.")
    
    # Scale the value to the interval [0, 1]
    normalized_value = (value - min_val) / (max_val - min_val)
    
    return normalized_value

def denormalize(value, min, max):

    return min + value*(max-min)


#test
def main():
    # print(disturbance(3,4))
    min, max, z, t, total_timesteps, p = 100,400,0.1,1,360,60
    # a = uniform_demand(min,max)
    # # print(a)
    # a = stochastic_demand(min, max, z,t, total_timesteps, 200, p)
    x = [100.0, 225.0]
    ratio = 1.25
    x = [ratio*i for i in x]
    # a = normalize(175,0,400)
    a = [normalize(i, 0,400) for i in x]
    print(a)
    # a = normalize(a,-1,1)
    # print(a)
    # a= denormalize(a,-1,1)
    # print(a)
    # print(int(denormalize(a,0,400)))
    print([denormalize(i, 0, 400) for i in a])



if __name__ == "__main__":
    main()