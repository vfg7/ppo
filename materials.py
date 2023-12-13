
class Material:
    #classe define as ordens relativas a um certo material

    def __init__(self,supply = None, demand = None, lead_time = None, cost=None):

        self.supply=supply #oferta deste material para um dado nó
        self.demand = demand #demanda deste material para um dado nó
        self.amount_on_node = 0 #quantidade de material disponível
        self.cost = cost #custo do material 
