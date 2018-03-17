import numpy as np
from random import random


def eps_greedy(env, Q, epsilon, session=None):
    return np.argmax(Q([env.current_state], session=session)) if random() <= epsilon else env.action_space.sample()
