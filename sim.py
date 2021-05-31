from random import seed
from random import random
seed(1)

# 1 day ~ sin(0.02x + 5) * 10 + 30
import math
def sim_curve(x, phaseshift, amp):
    # phaseshift default -/+ 1/2 pi
    pi = math.pi
    period = (2 * pi) / 365 
    y = math.sin(period * x + phaseshift) * 10 + amp
    #print(y)
    return y

def sim(x, range_, phaseshift, amp):
    y = sim_curve(x, phaseshift, amp)
    min_ = y - range_
    max_ = y + range_
    rand = random()
    rand_y = min_ + (rand * (max_ - min_))
    #print(max_)
    #print(min_)
    return rand_y

def sim_func_single(x, range_):
    min_ = x - range_
    max_ = x + range_
    rand = random()
    rand_y = min_ + (rand * (max_ - min_))
    #print(max_)
    #print(min_)
    return rand_y