import numpy as np
import collections

from numpy.core.fromnumeric import argmax
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
import flappy_bird_gym
import time
import math
import json
from utils.utils import *

def test(Q, traject):
    env = flappy_bird_gym.make("FlappyBird-cust-v0")
    
    obs = env.reset()
    state = obs2state(obs)
    done = False
    
    while not done:
        try:
            action = np.argmax(Q[state])
        except:
            action = env.action_space.sample()
            
        next_obs, reward, done, info = env.step(action) #two cases, reward always 1, reward always 0
        
        env.render()
        time.sleep(1/60)
        
        next_state = obs2state(next_obs)
        state = next_state
        traject.append(action)
        
    print('Achieved score: ', info['score'])
    env.close()

if __name__ == "__main__":
    with open("./src/records/Q.json") as f:
        Q = json.load(f)
    
    traject = []    
    test(Q, traject)
    print(traject)