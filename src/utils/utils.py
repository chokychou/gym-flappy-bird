import math

def obs2state(obs, multiplier=1000):
    x_pos = int(math.floor(obs[0]*multiplier))
    y_pos = int(math.floor(obs[1]*multiplier))
    y_vel = y_vel = int(obs[2])
    state_string = str(x_pos) + '_' + str(y_pos) + '_' + str(y_vel)

    return state_string