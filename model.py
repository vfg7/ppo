import random
import chain_nodes
from uncertainties import *


node_state = {'id':None, 'hierarchy': None,'current_stock':None,'current_ins':None,'future_ins':0}
state = {'timestep': None, 'demand': None} #a key for each id
# general state = {'id':A, 'state': node_state}
#estado, 27 dimensões
# estoque de cada nó (8) 
# produçao atual (disponível em t+1), produção esperada futura (agendada pra t > t+1), (suppliers) (9 ~ 12)
# entradas atuais e entradas esperadas futuras (todos os outross) (13~24)
# demandas por retailer (input) (25,26)
# timesteps restantes (27)
#estado: [estoque_n1, in_n1, future_in_n1, .._n8, demanda_1, ..._2, timesteps sobrando]

lmax = 2
max_demand = 400
max_timestep = 360

def calculate_state(chain, timestep, demand=[], normalized=None):
    
    state = []
    normalized_state =[]
    cumulative_stock = [0,0,0,0]
    for h in range(4):
        for node in chain:
            if node.hierarchy == h:
                if h==0:
                    in_cap = node.production_ratio
                else:
                    in_cap = cumulative_stock[h-1]

                stock_cap = node.stock_capacity
                cumulative_stock[h] = cumulative_stock[h] + stock_cap

                stock = node.current_stock
                ins = node.ins
                future_ins = node.future_ins
            
                my_state = [stock, ins, future_ins]
                state.extend(my_state)

                if normalized:
                    n_stock = normalize(stock, 0, stock_cap)
                    n_ins = normalize(ins, 0, in_cap)
                    n_future_ins = normalize(future_ins, 0, in_cap*(lmax-1))

                    n_state = [n_stock, n_ins, n_future_ins]
                    normalized_state.extend(n_state)


        
    state.extend(demand)
    state.append(timestep)

    if normalized:
        n_demand = [normalize(d, 0,400) for d in demand]
        n_timestep = normalize(timestep, 0, 360)
        normalized_state.extend(n_demand)
        normalized_state.append(n_timestep)

        return state, normalized_state
    else:
        return state

def set_state(chain, state):
    
    for node in chain:
        counter = node.hierarchy
        modifier = node.id

        node.current_stock = state[6*counter+3*modifier]
        node.ins = state[(6*counter+3*modifier)+1]
        node.future_ins = state[(6*counter+3*modifier)+2]    

    return chain            
        
#action, 14 dimensões
#material a ser produzido em cada supplier (2)
# para cada nó (Exceto retailers), material a ser entregue para os outros dois nós (3~14)
# action =[produzir_1, produzir_2, entrega_121, entrega_122]

# Regarding decisions on how much to produce on each supplier, an
# action value is multiplied by the supplier’s capacity. 
stock_cap_supplier = 6400
stock_cap = 1800
# About the decisions related to how # much to deliver, let anm and ano be the action values representing the amount of material to
# be delivered from a node n to its successor nodes m and o. If the node n is not a factory,
# the action values are ﬁrst multiplied by the node’s current stock levels Sin

def execute_action(action_vector, timestep, chain):
    #action vector comes normalized from policy, with all values between 0 and 1
    produce = action_vector[0:2] #hierarquia 0
    deliver = action_vector[2:]#0123, #4567 #891011
    print(deliver)
    carry = [0,0]

    for h in range(4):
        for node in chain:
            if node.hierarchy == h:
                if node.hierarchy ==0:
                    if node.id ==0:
                        prod = produce[0] * node.stock_capacity
                        d0 = deliver[0] * node.current_stock
                        d1 = deliver[1] * node.current_stock
                        from_one = node.update_state(prod,d0,0,d1,1)

                    else:
                        prod = produce[1] * node.stock_capacity
                        d0 = deliver[2] * node.current_stock
                        d1 = deliver[3] * node.current_stock
                        from_two =node.update_state(prod,d0,0,d1,1)
                else:
                    if node.current_stock < node.production_ratio:
                        ratio = node.current_stock
                    else:
                        ratio = node.production_ratio

                        carry = [x * ratio for x in carry]

                    if node.hierarchy ==3:

                        if node.id ==0: 
                            from_one = node.update_state(carry[0]) #carry sao entradas neste timestep
                        else:
                            from_two =node.update_state(carry[1])
                    else:
                        counter = node.hierarchy * 4
                        if node.id ==0:
                            from_one = node.update_state(carry[0],deliver[counter],0,deliver[counter+1],1)
                        else:
                            from_two =node.update_state(carry[1],deliver[counter+2],0,deliver[counter+3],1)
                try:
                    carry[0] = from_one[0] + from_two [0]
                    carry[1] = from_one[1] + from_two [1]
                except:
                    print('wrong order')

            
    timestep = timestep -1

    return chain

def penalty(chain):

    sai_que_e_tua = 0

    for node in chain:
        sai_que_e_tua = sai_que_e_tua + node.penalty
    
    return sai_que_e_tua


def operating_cost(chain):
    a=0
    b=0

    for node in chain:

        nodestock = node.current_stock * node.stock_cost
        nodeprocessing = node.production_cost * node.ins

        a = a+ nodestock + nodeprocessing + node.penalty

        futureprod = node.production_cost * node.future_ins

        out = 0
        try:
            for outs in node.outs:
                out = out + outs[0]+outs[1]
        except:
            out = 1

        logistics = node.logistic_cost * out

        b = b + futureprod + logistics
    
    return a + b


def generate_random_states(num_samples, normalized=None):
    states_list = []
    n_states =[]

    for _ in range(num_samples):
        state = np.zeros(27)

        # Set values for chain nodes 0 and 1 (suppliers)
        state[0] = np.random.randint(0, 6401)  # Stock for chain node 0
        state[1] = np.random.randint(0, 601)   # Current supply for chain node 0
        state[2] = np.random.randint(0, 601)   # Future supply for chain node 0
        state[3] = np.random.randint(0, 6401)  # Stock for chain node 1
        state[4] = np.random.randint(0, 601)   # Current supply for chain node 1
        state[5] = np.random.randint(0, 601)   # Future supply for chain node 1

        if normalized:
            n_state = np.zeros(27)
            n_state[0] = normalize(state[0],0, 6401)  # Stock for chain node 0
            n_state[1] = normalize(state[1],0, 601)   # Current supply for chain node 0
            n_state[2] = normalize(state[2],0, 601)    # Future supply for chain node 0
            n_state[3] = normalize(state[3],0, 6401)  # Stock for chain node 1
            n_state[4] = normalize(state[4],0, 601)    # Current supply for chain node 1
            n_state[5] = normalize(state[5],0, 601)    # Future supply for chain node 1


        # Set values for other chain nodes
        for i in range(6, 24, 3):
            state[i] = np.random.randint(0, 1601)  # Stock for other chain nodes
            state[i + 1] = np.random.randint(0, 841)  # Current supply for other chain nodes
            state[i + 2] = np.random.randint(0, 841)  # Future supply for other chain nodes

            if normalized:
                n_state[i] = normalize(state[i],0, 1601)
                n_state[i+1] = normalize(state[i+1],0, 841)
                n_state[i+2] = normalize(state[i+2],0, 841)


        # Set values for demand
        state[24] = np.random.randint(0, 401)  # Demand value 1
        state[25] = np.random.randint(0, 401)  # Demand value 2

        # Set time steps
        state[26] = np.random.randint(0, 361)  # Time steps

        if normalized:
            n_state[24] = normalize(state[24],0, 401)
            n_state[25] = normalize(state[25],0, 401)
            n_state[26] = normalize(state[26],0, 361)

            n_states.append(n_state)


        states_list.append(state)


    if normalized:

        return states_list, n_states
    else:
        return states_list




    