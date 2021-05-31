import gym
import pandas as pd
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from entities import Entity

class MicroGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
                
        self.house1 = Entity(30, 30)
        self.house2 = Entity(60, 5)
        self.house3 = Entity(50, 30)
        self.company = Entity()
        
        self.steps_beyond_done = None
        
        self.action_space = spaces.Discrete(9)
        
        
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32), np.array([np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max],
                        dtype=np.float32), dtype=np.float32)
        
        # low = 0, 0, 0, 0 to high = inf, inf, inf, inf
        
        #df=pd.DataFrame({'Buys':['House1', 'House1', 'House2', 'House2'],'Sells':['House1', 'House2', 'House1', 'House2']})
        df=pd.DataFrame({'Buys':['House1', 'House1', 'House1', 'House2', 'House2', 'House2', 'House3', 'House3', 'House3'],
                         'Sells':['House2', 'House3', 'Company', 'House1', 'House3', 'Company', 'House1', 'House2', 'Company']})
        print(df)
        
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        day1, prod1, expense1, price1, money1  = self.house1.next_state()
        day2, prod2, expense2, price2, money2  = self.house2.next_state()
        day3, prod3, expense3, price3, money3  = self.house3.next_state()
        dayC, prodC, expenseC, priceC, moneyC  = self.company.next_state()
        
        house1_crEl = prod1 - expense1 # example = - 10
        house2_crEl = prod2 - expense2 # example = + 20
        house3_crEl = prod3 - expense3
        house1_money = money1
        house2_money = money2
        house3_money = money3
        Company_money = moneyC
        
        if action == 0:
            ##################### House 1 buys from house 2
            h1_buffer = abs(house1_crEl) # this is what h1 needs
            if h1_buffer > house2_crEl: # if the need is higher than what it can buy from h2, then buy all
                h1_buffer = house2_crEl
                
            house1_crEl = house1_crEl + h1_buffer # add what h1 needs to it to get 0
            house2_crEl = house2_crEl - h1_buffer # remove the purchased electricity from h2
            house2_money = money2 + abs(h1_buffer * price2)     # add money to h2 according to their sale price
            
        elif action == 1:
            ##################### House 1 buys from house 3
            h1_buffer = abs(house1_crEl)
            if h1_buffer > house3_crEl:
                h1_buffer = house3_crEl
                
            house1_crEl = house1_crEl + h1_buffer
            house3_crEl = house3_crEl - h1_buffer
            house3_money = money3 + abs(h1_buffer * price3)
        elif action == 2:
            ##################### House 1 buys from company
            h1_buffer = abs(house1_crEl)
                
            house1_crEl = house2_crEl + h1_buffer
            
            Company_money = moneyC + abs(h1_buffer * priceC)
            
            
        elif action == 3:
            ##################### House 2 buys from house 1
            h2_buffer = abs(house2_crEl)
            if h2_buffer > house1_crEl:
                h2_buffer = house1_crEl
                
            house2_crEl = house2_crEl + h2_buffer
            house1_crEl = house1_crEl - h2_buffer
            house1_money = money1 + abs(h2_buffer * price1)
        elif action == 4:
            ##################### House 2 buys from house 3
            h2_buffer = abs(house2_crEl)
            if h2_buffer > house3_crEl:
                h2_buffer = house3_crEl
                
            house2_crEl = house2_crEl + h2_buffer
            house3_crEl = house3_crEl - h2_buffer
            house3_money = money3 + abs(h2_buffer * price3)
        elif action == 5:
            ##################### House 2 buys from Company
            h2_buffer = abs(house2_crEl)
            house2_crEl = house2_crEl + h2_buffer
            Company_money = moneyC + abs(h2_buffer * priceC)
            
            
        elif action == 6:
            ##################### House 3 buys from House 1
            h3_buffer = abs(house3_crEl)
            if h3_buffer > house1_crEl:
                h3_buffer = house1_crEl
                
            house3_crEl = house3_crEl + h3_buffer
            house1_crEl = house1_crEl - h3_buffer
            house1_money = money1 + abs(h3_buffer * price1)    
                
        elif action == 7:
            ##################### House 3 buys from House 2
            h3_buffer = abs(house3_crEl)
            if h3_buffer > house2_crEl:
                h3_buffer = house2_crEl
                
            house3_crEl = house3_crEl + h3_buffer
            house2_crEl = house2_crEl - h3_buffer
            house2_money = money2 + abs(h3_buffer * price2)
            
        elif action == 8:
            ##################### House 3 buys from Company
            h3_buffer = abs(house3_crEl)
            house3_crEl = house3_crEl + h3_buffer
            Company_money = moneyC + abs(h3_buffer * priceC)
        else:
            print("ERROR, Action is not available")
            
            
        self.state = (house1_crEl, house2_crEl, house3_crEl, house1_money, house2_money, house3_money, Company_money)
        
        condition = bool(
            #house1_crEl and house2_crEl and house3_crEl < 0
            (house1_crEl >= 0) and (house2_crEl >= 0) and (house3_crEl >= 0) 
        )
        
        reward = 0.0
        
        if house1_crEl >= 0:
            reward += 1.0
        else:
            reward -= 0.0
        
        if house2_crEl >= 0:
            reward += 1.0
        else:
            reward -= 0.0
        
        if house3_crEl >= 0:
            reward += 1.0
        else:
            reward -= 0.0
        
        #if action != 2 or action != 5 or action != 8:
        #    reward +=1
        
#        if condition:
#            reward = 1.0         
#        else:            
#            reward = 0.0
            
        done = bool(dayC > 365)
        #print("DONE: ", done)
        return np.array(self.state), reward, done, {}
    
    def reset(self):
        self.house1.reset()
        self.house2.reset()
        self.house3.reset()
        self.company.reset()
        day1, prod1, expense1, price1, money1  = self.house1.next_state()
        day2, prod2, expense2, price2, money2  = self.house2.next_state()
        day3, prod3, expense3, price3, money3  = self.house3.next_state()
        dayC, prodC, expenseC, priceC, moneyC  = self.company.next_state()
        
        house1_crEl = prod1 - expense1 # example = - 10
        house2_crEl = prod2 - expense2 # example = + 20
        house3_crEl = prod3 - expense3
        house1_money = money1
        house2_money = money2
        house3_money = money3
        Company_money = moneyC
        
        self.steps_beyond_done = None
        
        self.state = (house1_crEl, house2_crEl, house3_crEl, house1_money, house2_money, house3_money, Company_money)
        return np.array(self.state)
    
    def render(self, mode='human'):
        ...
    
    def close(self):
        ...