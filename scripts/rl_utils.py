def calc_return(episode, discount=1.0):
  ret = 0.0
  for trans in reversed(episode):
    ret += discount * trans.reward

  return ret
