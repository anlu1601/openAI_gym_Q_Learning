#!/usr/bin/python

import sys
import getopt
import statistics as st
from agent import *
from environment import *
from model import *
import time
import math

def main(argv):
    input_ = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["input_="])
    except getopt.GetoptError:
        print("main.py -i <q> || <n>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("main.py -i <q> || <n>")
            sys.exit()
        elif opt in ("-i", "--input"):
            input_ = arg
        elif opt in ("-o", "--ofile"):
            input_ = arg
    print("Input file is: ", input)

    print("Initializing environment...")
    env = MicroGridEnv()
    
    print("Initializing agent...")
    agent_ = Agent(env)

    if(input_ == "q" or input_ == "Q"):
        print("Training...")
        scores = agent_.training_q_learning(1000)
        print("Predicting...")
        scores_pred = agent_.agent_predict_q()
    else:
        training_data = agent_.initial_population(5)
        print("Creating model...")
        model_ = Model(training_data)
        model_.create()
        print("Training...")
        hist = model_.train()
        modell = model_.get_model()
        print("Predicting...")
        scores = agent_.agent_predict(modell)
    
    print("Finished")
    
if __name__ == "__main__":
    main(sys.argv[1:])

