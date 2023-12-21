from chain_nodes import *
from materials import *
from model import *
from policy import *
from uncertainties import *
import matplotlib.pyplot as plt
import numpy as np

#parameters
q = 8
stock_cap_supplier = 6400
stock_cap = 1800
stock_cost = 1

production_cost_sup = 6
process_cost = 10
transport_cost = 2
#penalty_cost_supply = 10
penalty_cost_demand = 200

prod_cap = 600
proc_cap = 840
log_cap = 600

demand_max = 500
demand_min = 0
leadtime_max = 4
timesteps = 360
demand_mean = 200
freq = 0.1

state_size = 27
action_size = 14

def create_chain(q):

    chain = []
    for x in range(q):
        if x%2==0:
            p=1
        else:
            p=1.1
        if x <2:
            stock = stock_cap_supplier
            prod = prod_cap
            prod_c= production_cost_sup
        else:
            stock = stock_cap
            prod_c = process_cost
            prod = proc_cap

        node = Chain_Node(id=q%2, stock_capacity=stock*p, stock_cost=stock_cost,
                          logistic_ratio=log_cap, logistic_cost=transport_cost,penalty_cost=penalty_cost_demand,
                          production_ratio=prod*p, production_cost=prod_c*p, h = x//2)
        node.print()
        chain.append(node)
    
    return chain

def simulate_chain(chain, max_time):
    #gerar condições iniciais, incluindo os estocásticos
    p = random.randint(0,60)

    demand = stochastic_demand(demand_min, demand_max, freq, 0, max_time, demand_mean, p)
    states, n_states = generate_random_states(1, normalized=True)

    cost_history =[]
    # Build initial models
    policy_model = build_ppo2_model(state_size, action_size)
    value_model = build_ppo2_model(state_size, action_size)

    for epoch in range(max_time):
        # run chain and calculate state
        chain = set_state(chain, states[-1])
        # evaluate state of chain 
        cost = operating_cost(chain)
        cost_history.append(cost)

        # apply policy, take action 
        print(cost, 'go to policy')
        action, policy_model, value_model = execute_policy(policy_model, value_model, chain, states[-1],n_states[-1], 100)

        # update parameters
        timesteps = max_time - epoch
        p = random.randint(0,60)
        demand =[stochastic_demand(min=demand_min, max=demand_max, freq=freq,t=epoch,total_timesteps=max_time,mean=demand_mean, std=p)
                 for x in range(2)]
        print(demand)
        updated_chain = execute_action(action, timesteps, chain)
        new_state, new_state_n = calculate_state(updated_chain, epoch, demand, True)
        print('new_state!')
        states.append(new_state)
        n_states.append(new_state_n)
        #guarda e imprime estado da cadeia, resultado da atuação do agente
    
    return cost_history

def evaluate_results(timeseries):

    print("Historical Performance Values:")
    print(timeseries)

    plt.figure(figsize=(10, 6))
    plt.plot(timeseries, label='Performance Over Time', marker='o')
    plt.title('Historical Performance Over Time')
    plt.xlabel('Time')
    plt.ylabel('Performance Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    initial_value = timeseries[0]
    max_value = np.max(timeseries)
    min_value = np.min(timeseries)
    std_value = np.std(timeseries)
    mean_value = np.mean(timeseries)
    below_mean_percentage = np.mean(np.array(timeseries) < mean_value) * 100

    print("\nStatistics:")
    print(f"Initial Value: {initial_value}")
    print(f"Max Value: {max_value}")
    print(f"Min Value: {min_value}")
    print(f"Standard Deviation: {std_value}")
    print(f"Mean Value: {mean_value}")
    print(f"Percentage of Values Below Mean: {below_mean_percentage}%")


#test
def main():
    chain_size =8
    timesteps = 60
    a = create_chain(chain_size)
    timeseries = simulate_chain(a, timesteps)
    print("results")
    evaluate_results(timeseries)

if __name__ == "__main__":
    main()