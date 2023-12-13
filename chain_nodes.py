from materials import Material

class Chain_Node:
    def __init__(self, id = None, stock_capacity = None, stock_cost = None, logistic_ratio = None,
                 logistic_cost = None,  penalty_cost =None, production_ratio=None,
                   production_cost=None, h = None):

        self.id = id

        self.stock_capacity=stock_capacity #capacidade total do nó (fábrica ou loja de produzir/processar o material)
        self.stock_cost = stock_cost #custos de armazenamento por qtd unitária do material, assumidos uniformes
        self.current_stock=0

        self.production_cost = production_cost #custos de produção
        self.production_ratio = production_ratio #capacidade do nó (produtiva) por timestep 
        # ou capacidade de processamento
        self.ins = 0
        self.forecast_ins = 0

        self.logistic_cost = logistic_cost
        self.logistic_ratio = logistic_ratio
        self.outs = []

        # self.prod_avaliability = True #disponibilidade de produção. Caso o nó tenha demanda de produção maior que um timestep, 
        # #produção neste nó fica indisponível durante o tempo em que está produzindo 
        # self.log_availability = True #disponibilidade de entrega
        # self.times =[0,0] #supply and demand timestep overhead
        #disponibilidade vai ficar diretamente baseada no estoque
        self.on_stock = [0,0] #lista de produtos que podem estar no estoque deste nó
        #penalidades de produção/demanda
        self.penalty =0
        self.penalty_cost = penalty_cost
        #dados da cadeia de suprimento
        self.supply_chain = []
        self.hierarchy = h

    def print(self):
        my_dict ={}
        my_dict['id'] = self.id

        my_dict['stock_capacity']= self.stock_capacity #capacidade total do nó (fábrica ou loja de produzir/processar o material)
        my_dict['stock_cost']  = self.stock_cost  #custos de armazenamento por qtd unitária do material, assumidos uniformes
        my_dict['current_stock']  = self.current_stock

        my_dict['production_cost'] = self.production_cost  #custos de produção
        my_dict['production_ratio']  =  self.production_ratio #capacidade do nó (produtiva) por timestep 
            # ou capacidade de processamento
        my_dict['ins'] = self.ins 
        my_dict['forecast_ins'] = self.forecast_ins 

        my_dict['logistic_cost'] = self.logistic_cost 
        my_dict['logistic_ratio'] = self.logistic_ratio 
        my_dict['outs'] = self.outs 
        my_dict['on_stock']  = self.on_stock  #lista de produtos que podem estar no estoque deste nó
            #penalidades de produção/demanda
        my_dict['penalty'] =  self.penalty 
        my_dict['penalty_cost'] =   self.penalty_cost 
            #dados da cadeia de suprimento
        my_dict['supply_chain'] =   self.supply_chain 
        my_dict['hierarchy'] =   self.hierarchy 
        
        print(my_dict)
        return my_dict

# adiciona material a ser produzido
    def produce (self,  supply):
        #check stock
        free_stock = self.stock_capacity -  self.current_stock
        
        #check supply
        # supply = material.supply

        #check availability.
        # if not self.prod_avaliability:
        #     producing_timestep = self.times[0]
        #     if not producing_timestep <1:
        #         #mais de um timestep pra acabar a produção que tá sendo feito
        #         self.penalty = (supply) * self.penalty_cost
        #         self.times[0] = producing_timestep - 1
        #         return self.production_capacity, 0
        #     else: 
        #         #menos de um timestep pra acabar a produção que tá sendo feito, penalidade e produção proporcionais
        #         ratio = 1 - producing_timestep
        #         self.penalty = (supply) * self.penalty_cost * producing_timestep
        #         self.times[0] = 0

        #         supply = supply * ratio
  
        #if avaliable,
        if supply > free_stock:
            if not self.hierarchy == 0:
                self.penalty = self.penalty + (supply - free_stock) * (self.penalty_cost/20)
            #else, não é produzido, mas recebido
            produced = free_stock
        else:
            produced = supply

        #check delays
        # delay = material.lead_time #apply function? this value must be normalized  ~ PPO
        # produced =  produced * delay

        # #update stock
        # if material not in self.on_stock:
        #     material.amount_on_node = produced
        #     self.on_stock.append(material)
        # else:
        #     i = self.on_stock.index(material)
        #     self.on_stock[i].amount_on_node = self.on_stock[i].amount_on_node + produced

        return produced
    
    def check_storage(self, material = None):
        #quantidade produzida neste timestep é comparada com a disponibilidade 
        try:

            if material:
                if material in self.on_stock:
                    i = self.on_stock.index(material)
                    return self.on_stock[i].amount_on_node
                else:
                    return 0

            else:
                if self.on_stock == None:
                    capacity = self.stock_capacity
                else:
                    for m in self.on_stock:
                        partial_capacity = partial_capacity + m.amount_on_node

                    capacity = self.stock_capacity - partial_capacity
            
            return capacity
        
        except Exception:
            print(Exception)
            pass
        
    def deliver (self, demand):
        #check stock
        stock = self.current_stock
        if stock == 0 :
            return 0, 0
        
        #check demand
        # demand = material.demand

         #check availability.
        # if not self.log_availability:
        #     deliver_timestep = self.times[1]
        #     if not deliver_timestep <1:
        #         #mais de um timestep pra acabar a entrega que tá sendo feito
        #         self.penalty = (demand) * self.penalty_cost
        #         self.times[1] = deliver_timestep - 1
        #         return self.logistic_capacity, 0
        #     else: 
        #         #menos de um timestep pra acabar a produção que tá sendo feito, penalidade e produção proporcionais
        #         ratio = 1 - deliver_timestep
        #         self.penalty = (demand) * self.penalty_cost * deliver_timestep
        #         self.times[1] = 0
        #         demand = demand * ratio

        if demand > stock:
            self.penalty = self.penalty + (demand - stock) * self.penalty_cost
            delivery = stock
        elif demand > self.logistic_ratio:
            self.penalty = self.penalty + (demand - stock) * self.penalty_cost
            delivery = self.logistic_ratio
        else:
            delivery = demand
        
        #compute uncertain delay
        # delay = logistic_lead_time #uncertain?
        # delivery = delivery * delay

        # #update stock        
        # i = self.on_stock.index(material)
        # new_amount = self.on_stock[i].amount_on_node - delivery

        # if new_amount == 0:
        #     self.on_stock.remove(material)
        # else:
        #     self.on_stock[i].amount_on_node = self.on_stock[i].amount_on_node - delivery

        return delivery

    def count_me_stock(self):
        for mats in self.on_stock:
            self.current_stock = self.current_stock + mats.amount_on_node


    def supply_chain_manager(self, supply_future = None, supply_now = None, demand_one = None, 
                             id_one = None, demand_two = None, id_two = None):
        #what happens in this time step, stays in this timestep... or not
        #imaginando o timestep como um intervalo contínuo (tipo, um dia)
        #por padrão, toda produção começa no timestep e toda entrega começa no final do timestep 
        # (a ser entregue para o próximo nó no começo do timestep seguinte. Isto é, vira o supply do outro)
        self.forecast_ins = self.produce(supply_future) #foi produzido neste timestep
        self.ins = self.produce(supply_now) #foi produzido neste timestep

        delivery_hierarchy = self.hierarchy + 1
        one = self.deliver(demand_one) #será entregue para o nó seguinte no próximo timestep
        outs_to_one = {'id': id_one, 'hierarchy':delivery_hierarchy,'amount':one}
        two = self.deliver(demand_two) #será entregue para o nó seguinte no próximo timestep
        outs_to_two = {'id': id_two, 'hierarchy':delivery_hierarchy,'amount':two}
        self.outs= [outs_to_one, outs_to_two]
        #tanto faz o quanto eu vou entregar, oq ue vou entregar é o que importa?

        self.count_me_stock()



