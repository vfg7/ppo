from chain_nodes import *
from materials import *
# from model import *
# from policy import *
from uncertainties import *

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

        node = Chain_Node(id=q, stock_capacity=stock*p, stock_cost=stock_cost,
                          logistic_ratio=log_cap, logistic_cost=transport_cost,penalty_cost=penalty_cost_demand,
                          production_ratio=prod*p, production_cost=prod_c*p, h = x//2)
        node.print()
        chain.append(node)
    
    return chain

def simulate_chain(chain, timesteps):
    #gerar condições iniciais, incluindo os estocásticos
    
    for epochs in range(timesteps):
        # run chain and calculate state
        # evaluate state of chain 
        # apply policy, take action 
        # update parameters
        #guarda e imprime estado da cadeia, resultado da atuação do agente
        #
        break


    return True

#test
def main():

    a = create_chain(2)
    simulate_chain(a)

if __name__ == "__main__":
    main()