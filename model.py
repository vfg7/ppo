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

def calculate_state(chain, timestep, demand=[]):
    
    state = []
    for node in chain:
        stock = node.current_stock
        ins = node.ins
        future_ins = node.future_ins
        my_state = [stock, ins, future_ins]

        state.extend(my_state)
    
    state.extend(demand)
    state.append(timestep)

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
def execute_action(action_vector, demand, timestep, chain):
    produce = action_vector[0:1]
    deliver = action_vector[2:]#0123, #4567 #891011

    carry = []
    for node in chain:
        if node.hierarchy ==0:
            if node.id ==0:
                from_one = node.update_state(produce[0],deliver[0],0,deliver[1],1)
            else:
                from_two =node.update_state(produce[1],deliver[2],0,deliver[3],1)
        elif node.hierarchy ==3:
            if node.id ==0:
                from_one = node.update_state(carry[0],demand[0])
            else:
                from_two =node.update_state(carry[1],demand[1])
        else:
            counter = node.hierarchy * 4
            if node.id ==0:
                from_one = node.update_state(carry[0],deliver[counter],0,deliver[counter+1],1)
            else:
                from_two =node.update_state(carry[1],deliver[counter+2],0,deliver[counter+3],1)

        carry[0] = from_one[0] + from_two [0]
        carry[1] = from_one[1] + from_two [1]

        
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
        for outs in node.outs:
            out = out + outs[0]+outs[1] #rever estrutura de dados daqui 

        logistics = node.logistic_cost * out

        b = b + futureprod + logistics
    
    return a + b


def generate_random_states(num_samples):
    states_list = []

    for _ in range(num_samples):
        state = np.zeros(27)

        # Set values for chain nodes 0 and 1 (suppliers)
        state[0] = np.random.randint(0, 6401)  # Stock for chain node 0
        state[1] = np.random.randint(0, 601)   # Current supply for chain node 0
        state[2] = np.random.randint(0, 601)   # Future supply for chain node 0
        state[3] = np.random.randint(0, 6401)  # Stock for chain node 1
        state[4] = np.random.randint(0, 601)   # Current supply for chain node 1
        state[5] = np.random.randint(0, 601)   # Future supply for chain node 1

        # Set values for other chain nodes
        for i in range(6, 24, 3):
            state[i] = np.random.randint(0, 1601)  # Stock for other chain nodes
            state[i + 1] = np.random.randint(0, 841)  # Current supply for other chain nodes
            state[i + 2] = np.random.randint(0, 841)  # Future supply for other chain nodes

        # Set values for demand
        state[24] = np.random.randint(0, 401)  # Demand value 1
        state[25] = np.random.randint(0, 401)  # Demand value 2

        # Set time steps
        state[26] = np.random.randint(0, 361)  # Time steps

        states_list.append(state)

    # Convert the list of states to a NumPy array
    states_array = np.array(states_list)

    return states_array



    