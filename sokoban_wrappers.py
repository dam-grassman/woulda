#############################################
############# SOKOBAN WRAPPER ###############
#############################################

import gym
import gym_sokoban
import numpy as np
from gym import spaces
from gym import ObservationWrapper

class ActionWrapper(gym.ActionWrapper):

    """ Limit the number of actions to 5 by removing the redundancy of
    the last three original actions """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = spaces.Discrete(5)
        
    def get_action_meanings(self):
        dic = self.env.get_action_meanings()
        len_keys = len(dic.keys())
        for key in range(5, len_keys):
            del dic[key]
        return dic 
    
    def action(self, act):
        return act

class LengthWrapper(gym.Wrapper):

    """ Optional Wrapper to limit the game at 50 steps"""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.length = 0 
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.length = self.length + 1
        if done == True:
            self.length = 0
        if self.length == 50 :
            done = True
            self.length = 0
        reward += 0.1
        return obs, reward, done, info

Element = {0 : [160, 45,   45],
           1 : [0,     0,   0], 
           2 : [254, 126, 125], 
           4 : [142, 121,  56], 
           3 : [254,  95,  56], 
           5 : [160, 212,  56], 
           6 : [219, 212,  56]}

Encoder = {k: np.array([0. if i!=k else 1. for i in range(7)]) for k in range(7)}

def render_po(po_array):

    """ Convert PO-Sokoban obs into a colored image """

    def to_colors(arr):
        if np.max(arr) == 1:
            return Element[np.argmax(arr)]
        else : 
            return [255, 255, 255]
    return np.apply_along_axis(to_colors, -1, po_array)

def render_full(full_array):
    
    """ Convert Sokoban fully observed state into a colored image """

    H, W = full_array.shape[:2]
    def to_colors(arr):
            return Element[arr[0]]
    return np.apply_along_axis(to_colors, -1, full_array.reshape(H,W,1))



class PartialObservation(ObservationWrapper):

    """ Wrapper to transform the Sokoban game into a partial observed environment.
    The agent won't observe the entire state but only a partial board whose cells are masked
    by iid noise with a certain probability (default p=0.9)."""
    
    def __init__(self, env, mask_proba=0.9):
        super(PartialObservation, self).__init__(env)
        self.mask_proba = mask_proba
        self.env = env
        H, W = env.dim_room 
        self.observation_space = spaces.Box(0,1, shape =  (H, W, 7), dtype=np.float32)
        self.agent_pos = None

    def observation(self, observation):
        """ Return the partial observed state """
        masked_observation = self._mask_observation(observation)
        return masked_observation
 
    def _mask_observation(self, observation):
        
        h, w, c = self.observation_space.shape
        observation = np.apply_along_axis(lambda a : Encoder[a[0]], -1, self.env.room_state.reshape(h, w, 1))
 
        agent_pos = (np.max(np.argmax(observation[:,:,5:], axis=0)), 
                     np.max(np.argmax(observation[:,:,5:], axis=1)))

        mask = np.random.random(int(h*w))>=(self.mask_proba)
        mask = mask.reshape((h, w))       
                    
        mask[max(0,(agent_pos[0]-1)):min(h,(agent_pos[0]+2)), 
            max(0,(agent_pos[1]-1)):min(w,(agent_pos[1]+2))] =  True

        observation = observation * np.stack([mask for i in range(7)], axis=2)
        return observation

def convert_room(room):
    h, w = room.shape
    return np.apply_along_axis(lambda a : Encoder[a[0]], -1, room.reshape(h, w, 1))

class SwitchAxis(ObservationWrapper):

    """ Wrapper to transform the Sokoban game into a partial observed environment.
    The agent won't observe the entire state but only a partial board whose cells are masked
    by iid noise with a certain probability (default p=0.9)."""
    
    def __init__(self, env):
        super(SwitchAxis, self).__init__(env)
        self.env = env
        H, W, c = self.env.observation_space.shape
        self.observation_space = spaces.Box(0,1, shape =  (c, H, W), dtype=np.float32)

    def observation(self, observation):
        """ Return the partial observed state """
        return  self.env.observation(observation).transpose(2,0,1)
 
# Registration
from gym.envs.registration import register
try :
    register(id='Sokoban-allow-reset-v0', entry_point='gym_sokoban.envs:SokobanEnv1', kwargs={'reset':False} )
except : 
    pass
try :
    register(id='Sokoban-small-allow-reset-v0', entry_point='gym_sokoban.envs:SokobanEnv_Small0', kwargs={'reset':False} )
except : 
    pass
to_room_fixed = np.vectorize(lambda el : 1 if el>2 else el)

class CustomReset(gym.Wrapper):

    """ ALlow the Sokoban enviroment to reset with 
    a given initial state (which is not feasible with 
    the default env) """
    
    def __init__(self, env):
        super(CustomReset, self).__init__(env)
        self.env = env
        self.env.room_fixed = None
        self.env.room_state = None
        self.env.box_mapping = None
        
    def reset(self, array = np.array([]), second_player=False, render_mode='rgb_array'):
        if array.size == 0 :
            try:
                self.room_fixed, self.room_state, self.box_mapping = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    second_player=second_player
                )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                return self.reset(second_player=second_player)
        
        else : 
            #print('Initial scenario provided')
            self.env.room_state = array
            self.env.room_fixed = to_room_fixed(array)
            self.env.box_mapping = {}
            
        self.env.player_position = np.argwhere(self.room_state == 5)[0]
        self.env.num_env_steps = 0
        self.env.reward_last = 0
        self.env.boxes_on_target = 0

        starting_observation = self.env.render(render_mode)
        return starting_observation

room_buffer = np.load('data/rooms7x7.npy')
#room_buffer = room_buffer[:120]

class ResetfromBuffer(gym.Wrapper):

    """ Wrapper that allows the environment to reset by drawing 
    its initial state from a Buffer """

    def __init__(self, env, nb_room = room_buffer.shape[0]):
        super(ResetfromBuffer, self).__init__(env)
        self.env = env
        self.rand = -1
        self.nb_room = nb_room
        self.nb_room_old = nb_room

    def reset(self, provide_rand=-1, **kwargs):

        if 'array' in kwargs.keys():
            return self.env.reset(**kwargs)

        if self.nb_room_old != self.nb_room : 
            print('Nb of rooms changed', self.nb_room)
            self.nb_room_old = self.nb_room

        if provide_rand == -1:
            self.rand = np.random.randint(self.nb_room)
        else : 
            self.rand = provide_rand

        array = room_buffer[self.rand].copy()
        return self.env.reset(**{'array':array})


def make_sokoban(env_name='Sokoban-v0', 
                 action_wrapper=True, 
                 po_wrapper=True, 
                 length_wrapper=True, 
                 allow_reset=False,
                 mask_proba=0.9):
    
    """ Build and Wrap the Sokoban Env """

    if allow_reset :
        try :
            env_name = env_name.split('-')[:-1] + ['allow-reset'] + env_name.split('-')[-1:]
            env_name = str.join('-', env_name)
            env = gym.make(env_name)
            env = CustomReset(env)
        except : 
            print('{} not registered..'.format(env_name))
            return 
    else :
        env = gym.make(env_name)
    if action_wrapper:
        env = ActionWrapper(env)
    if po_wrapper :
        env = PartialObservation(env, mask_proba)
    if length_wrapper :
        env = LengthWrapper(env)
    return env

