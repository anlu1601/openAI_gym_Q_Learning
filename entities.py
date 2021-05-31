import math
from sim import *

class Entity:
    
    def __init__(self, amplitude_prod = 30, amplitude_ex = 30):
        pi = math.pi
        self.day = 1
        self.amp_prod = amplitude_prod
        self.amp_ex = amplitude_ex
        self.production = sim(self.day, 5, -1/2 * pi, amplitude_prod)
        self.expense = sim(self.day, 5, 1/2 * pi, amplitude_ex)
        self.saleprice = sim_func_single(0.4, 0.1)
        self.money = 0
        
    def next_state(self):
        pi = math.pi
        prod = self.production
        day = self.day
        expense = self.expense
        price = self.saleprice
        self.day = day + 1
        self.production = sim(self.day, 5, -1/2 * pi, self.amp_prod)
        self.expense = sim(self.day, 5, 1/2 * pi, self.amp_ex)
        self.saleprice = sim_func_single(0.4, 0.1)
        #print("Day: ",day)
        #print("KWH produced: ", prod)
        #print("KWH used: ", expense)
        #print("Sale price: ", price)
        ret = [day, prod, expense, price, self.money]
        #print(ret)
        return ret
    def reset(self):
        self.day = 1