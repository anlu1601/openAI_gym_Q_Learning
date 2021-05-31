from statistics import median, mean
from collections import Counter
import numpy as np
import time
import random

class Agent():

    def __init__(self,env):
        self.env = env
        self.latency = []
        self.accepted_scores = []
        self.q_table = np.zeros([8, env.action_space.n])
    
    def encode(self, x, size):
        output = []
        for i in range(size):
            if i == x:
                output.append(1)
            else:
                output.append(0)
        return output
    
    def state_decoder(self, state):
        
        first = state[0]
        second = state[1]
        third = state[2]
        
        first = 1 if first >= 0 else 0
        second = 1 if second >= 0 else 0
        third = 1 if third >= 0 else 0
        
        combined = str(first) + str(second) + str(third)
        state_int = int(combined, 2)
        
        return state_int

    def training_q_learning(self, iterations):
        scores = []
        q_table = self.q_table
        # Hyperparameters
        alpha = 0.0001
        gamma = 0.01
        epsilon = 0.1
        
        for _ in range(iterations):
            score = 0
            state = self.env.reset()
            state_code = self.state_decoder(state)
            
            for _ in range(365):
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(q_table[state_code])
                
                observation, reward, done, info = self.env.step(action)
                
                observation_code = self.state_decoder(observation)
                
                old_value = q_table[state_code, action]
                next_max = np.max(q_table[observation_code])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state_code, action] = new_value
                
                score += reward
                
            scores.append(score)
            
        self.q_table = q_table
            
        print('Average score:',mean(scores))
        print('Median score for scores:',median(scores))
        print('Maximum score:', 365*3)
        print(Counter(scores))
        return scores
    
    def agent_predict_q(self):
        scores = []
        choices = []
        latency_all = []
        q_table = self.q_table
        print("Starting prediction...")
        for each_game in range(10):
            score = 0
            game_memory = []
            prev_obs = []
            latency_year = []
            state = self.env.reset()
            print("Year ", each_game)
            for _ in range(365):
                
                #print("year")
                t = time.time()

                
                action = np.argmax(q_table[self.state_decoder(state)])

                choices.append(action)

                new_observation, reward, done, info = self.env.step(action)

                
                #print(new_observation)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                #print(score)
                score+=reward
                elapsed = time.time() - t
                latency_year.append(elapsed)
                
                if done: break

            scores.append(score)
            latency_all.append(latency_year)

        print('Average score:',sum(scores)/len(scores))
        print('Maximum score:', 365*3)
        print(scores)
        print('choice 0:{}  choice 1:{} choice 2:{} choice 3:{} choice 4:{} choice 5:{} choice 6:{} choice 7:{} choice 9:{}'.
              format(choices.count(0)/len(choices),choices.count(1)/len(choices),
                     choices.count(2)/len(choices),choices.count(3)/len(choices),
                     choices.count(4)/len(choices),choices.count(5)/len(choices),choices.count(6)/len(choices),choices.count(7)/len(choices),choices.count(8)/len(choices)))
        self.latency = latency_all
        return scores
    
    
    def initial_population(self, iterations):
        # [OBS, MOVES]
        training_data = []
        # all scores:
        scores = []
        # just the scores that met our threshold:
        accepted_scores = []
        # iterate through however many games we want:
        for _ in range(iterations):
            score = 0
            # moves specifically from this environment:
            observation_action_tuple = []
            # previous observation that we saw
            prev_observation = []
            # for each frame in 200
            for _ in range(365):
                # choose random action (0 or 1)
                action = self.env.action_space.sample()
                # do it!
                observation, reward, done, info = self.env.step(action)
                #print(action)
                # notice that the observation is returned FROM the action
                # so we'll store the previous observation here, pairing
                # the prev observation to the action we'll take.
                if len(prev_observation) > 0 :
                    observation_action_tuple.append([prev_observation, action])
                prev_observation = observation
                score += reward
                if done: break

            # IF our score is higher than our threshold, we'd like to save
            # every move we made
            # NOTE the reinforcement methodology here. 
            # all we're doing is reinforcing the score, we're not trying 
            # to influence the machine in any way as to HOW that score is 
            # reached.
            if score >= 986: #986
                accepted_scores.append(score)
                for data in observation_action_tuple:
                    # convert to one-hot (this is the output layer for our neural network)
                    output = self.encode(data[1], 9)
                    #output = data[1]
                    #print(data[1])

                    # saving our training data
                    training_data.append([data[0], output])

            # reset env to play again
            self.env.reset()
            # save overall scores
            scores.append(score)

        # just in case you wanted to reference later
        #training_data_save = np.array(training_data)
        #np.save('saved.npy',training_data_save)

        # some stats here, to further illustrate the neural network magic!
        if training_data:
            print('Average accepted score:',mean(accepted_scores))
            print('Median score for accepted scores:',median(accepted_scores))
            print('Maximum score', 365*3)
            print(Counter(accepted_scores))
        else:
            print('No acceptable data to show')
        self.accepted_scores = accepted_scores
        
        return training_data


    def agent_predict(self, model):
        scores = []
        choices = []
        latency_all = []
        print("init")
        for each_game in range(10):
            score = 0
            game_memory = []
            prev_obs = []
            latency_year = []
            self.env.reset()
            print("game")
            for _ in range(365):
                
                
                #print("year")
                t = time.time()

                if len(prev_obs)==0:
                    action = self.env.action_space.sample()
                else:
                    #print(prev_obs.shape)
                    action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs)))[0])

                choices.append(action)

                new_observation, reward, done, info = self.env.step(action)

                elapsed = time.time() - t

                latency_year.append(elapsed)
                #print(new_observation)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                #print(score)
                score+=reward
                if done: break

            scores.append(score)
            latency_all.append(latency_year)

        print('Average score:',sum(scores)/len(scores))
        print('Maximum score:', 365*3)
        print(scores)
        print('choice 0:{}  choice 1:{} choice 2:{} choice 3:{} choice 4:{} choice 5:{} choice 6:{} choice 7:{} choice 9:{}'.
              format(choices.count(0)/len(choices),choices.count(1)/len(choices),
                     choices.count(2)/len(choices),choices.count(3)/len(choices),
                     choices.count(4)/len(choices),choices.count(5)/len(choices),choices.count(6)/len(choices),choices.count(7)/len(choices),choices.count(8)/len(choices)))
        self.latency = latency_all
        return scores
    
    def get_latency(self):
        return self.latency
    
    def get_scores(self):
        return self.accepted_scores