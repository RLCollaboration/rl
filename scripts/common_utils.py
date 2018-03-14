import numpy as np
from random import random


def eps_greedy(env, epsilon, Q):
    return np.argmax(Q(env.current_state)) if random() <= epsilon else env.action_space.sample()
