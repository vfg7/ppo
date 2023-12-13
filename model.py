import random
import chain_nodes
from uncertainties import *
from policy import *

node_state = {'id':None, 'hierarchy': None,'current_stock':None,'current_ins':None,'future_ins':0}
state = {'timestep': None, 'demand': None} #a key for each id
# general state = {'id':A, 'state': node_state}
#estado, 27 dimensões
# estoque de cada nó (8) 
# produçao atual (disponível em t+1), produção esperada futura (agendada pra t > t+1), (suppliers) (9 ~ 12)
# entradas atuais e entradas esperadas futuras (todos os outross) (13~24)
# demandas por retailer (input) (25,26)
# timesteps restantes (27)

def state_of_node(state, chain, timestep, demand):
    state['timestep'] = timestep
    state['demand'] = demand

    for node in chain:
        if node.id in state.keys():
            this_node_state = {}
            this_node_state['current_stock'] = node.current_stock

            this_node_state['current_ins'] = node.ins
            this_node_state['future_ins'] = node.forecast_ins

            state[node.id] = this_node_state

#action, 14 dimensões
#material a ser produzido em cada supplier (2)
# para cada nó (Exceto retailers), material a ser entregue para os outros dois nós (3~14)
def action_in_chain(action_vector, chain):
    produce = action_vector[0:1]
    deliver_01 = action_vector[2:5]
    deliver_12 = action_vector[6:9]
    deliver_23 = action_vector[10:13]

    #chama os nós e coloca os inputs
    # supplier_a.produce(produce[0]) ex

    # if node.hierarchy == x:
        #node_x.deliver(deliver_xy[amount_a, amount_b])
    
    #update state, count timestep
    #return new_state



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


    