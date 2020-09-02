from sokoban_wrappers import *
import numpy as np

def rollout(T=50, mask_proba=0.9, array = None, **kwargs):

    """ Perform a rollout with a random policy """

    #Create the Sokoban env, and initialize history
    env = make_sokoban(mask_proba = mask_proba, **kwargs)
    history, rewards, actions = [], [], []
    
    #If an initial state is provided
    if array is not None :
        obs = env.reset(**{'array':array})
    else :
        obs = env.reset()
    initial_state = env.room_state.copy()
    
    # Rollout for T steps
    for t in range(T):
        history.append(obs)
        action = np.random.randint(env.action_space.n)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)

    return {'obs' : history,
            'rewards': rewards, 
            'actions': actions, 
            'initial_state':initial_state}
