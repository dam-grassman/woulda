# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
#from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore") 

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('--processes', default=10, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()

import gym
import gym_sokoban

import sys ; import os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_path + '/baby_a3c')
import atari_wrappers 

from sokoban_wrappers import *
from DRAW.draw_model import DRAWModel, train_draw, params
import torch.optim as optim


def create_env(env_name, nb_room = 5, mask_proba=0.7):
    return atari_wrappers.wrap_sokoban(
            env_name,
            clip_rewards=False,
            frame_stack=False,
            scale=False,
            allow_reset=True,
            nb_room = nb_room,
            mask_proba = mask_proba
        )


discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
prepro = lambda img: img.astype(np.float32)

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

class SokobanPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(SokobanPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.gru = nn.GRUCell(32 * 7 * 7, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        hx = self.gru(x.view(-1, 32 * 7 * 7), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = -1
        print('Paths', paths)
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
            print('loaded')
        print("\tno saved models") if step == -1 else print("\tloaded model: {}".format(paths[ix]))
        return max(0,step)

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

class SharedAdam2(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam2, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

def train(shared_model, shared_optimizer, shared_draw_model, shared_draw_optimizer, params, rank, args, info):

    nb_room = 100
    env = create_env(args.env, nb_room) # make a local (unshared) environment
    env.seed(args.seed + rank) ; torch.manual_seed(args.seed + rank) # seed everything
    model = SokobanPolicy(channels=7, memsize=args.hidden, num_actions=args.num_actions)
    draw_model = DRAWModel(params).to(params['device'])
    draw_optimizer = optim.Adam(draw_model.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))

    state = prepro(env.reset()) # get first state
    #history = [np.expand_dims(state, axis=0)]
    state = torch.tensor(state) # get first state
    print('State', state.shape)

    true_initial_states = [env.room_state.copy()]
    history = []
    rollout = []
    start_time = last_disp_time = time.time()
    last_disp_time_save = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    counterfactual = False
    counterfactual_counter = 10
    draw_online = True

    while info['frames'][0] <= 8e7 or args.test: # openai baselines uses 40M frames...we'll use 80M
        

        model.load_state_dict(shared_model.state_dict()) # sync with shared model

        hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(args.rnn_steps):

            episode_length += 1
            value, logit, hx = model((state.view(1,7,7,7), hx))
            logp = F.log_softmax(logit, dim=-1)

            if true_initial_states != []:
                history.append(np.expand_dims(state, axis=0))

            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            state, reward, done, _ = env.step(action.numpy()[0])
            
            if args.render: 
                time.sleep(0.2)
                env.render()

            state = torch.tensor(prepro(state)) ; epr += reward
            done = done or episode_length >= 50 # don't playing one ep for too long
            
            if done and true_initial_states != [] :
                #print('DONE History', len(history))
                if len(history) > 52 :
                    print('Weird:', len(history) )
                
                if len(history) > 50 :
                    history = history[:50]
                else :
                    while len(history) < 50 :
                        history.append(history[-1])
                rollout.append(np.expand_dims(np.concatenate(history, axis = 0), axis=0))
                history = []

            info['frames'].add_(1) ; num_frames = int(info['frames'].item())
            if rank == 0 and time.time() - last_disp_time_save > 5*60: # save every 5 Min
                printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))
                torch.save(shared_optimizer.state_dict(), args.save_dir+'optimizer.{:.0f}.tar'.format(num_frames/1e6))
                last_disp_time_save = time.time()   

            if done: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)
                

            timelasp =  time.time() - last_disp_time

            if timelasp > 60: # print info ~ every minute
                if rank == 0 :
                    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                        .format(elapsed, info['episodes'].item(), num_frames/1e6,
                        info['run_epr'].item(), info['run_loss'].item()))
                
                last_disp_time = time.time()

                if info['run_epr'].item() > 6:
                    nb_room += 5 #increase the number of room in the buffer
                    nb_room = min(nb_room, 100)
                    env.nb_room = nb_room
                    env.env.nb_room = nb_room

            if done: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(prepro(env.reset()))
                true_initial_states.append(env.room_state.copy())

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        next_value = torch.zeros(1,1) if done else model((state.unsqueeze(0), hx))[0]
        values.append(next_value.detach())

        loss = cost_func(args, torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()   
        shared_optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        if draw_online :

            if len(rollout) > 200 : 
                print('Starting DRAW training :')
                rollout = np.concatenate(rollout, axis=0)
                #rollout = np.transpose(rollout, (0, 1, 4, 2, 3))
                true_initial_states = true_initial_states[:rollout.shape[0]]
                true_initial_states = np.concatenate([np.expand_dims(convert_room(ist), axis=0) for ist in true_initial_states], axis = 0)
                true_initial_states = np.expand_dims(true_initial_states, axis=1)
                true_initial_states = np.transpose(true_initial_states, (0, 1, 4, 2, 3))
                dataloader_train = torch.utils.data.DataLoader(np.concatenate([true_initial_states, rollout], axis=1), 
                                                batch_size=4, shuffle=True)

                print('Data to train on', np.concatenate([true_initial_states, rollout], axis=1).shape)
                print('LEN', len(dataloader_train))
                #train_draw(draw_model, draw_optimizer, params, dataloader_train, None, folder_checkpoint = 'DRAW/checkpoint2')
                
                losses = []
                iters = 0
                avg_loss = 0

                for epoch in range(0, params['epoch_num']):    
                    draw_model.load_state_dict(shared_draw_model.state_dict()) # sync with draw shared model
                    avg_loss = 0            
                    for i, data_batch in enumerate(dataloader_train, 0):

                        #print('Batch', data_batch.shape)
                        data = torch.tensor(data_batch[:,1:,:,:,:], dtype = torch.float32).to(params['device'])
                        data_tr = torch.tensor(data_batch[:,0], dtype = torch.float32).to(params['device'])
                        #print('TEST')
                        #print(data_tr[0,:].argmax(0))
                        #print(data[0,0].argmax(0))
                        #print(data[0,0].sum(0))

                        #break
                        # Get batch size.
                        bs = data.shape[0]
                        data = data.to(params['device'])
                        draw_optimizer.zero_grad()
                        
                        # Calculate the loss.
                        draw_loss = draw_model.loss(data, data_tr)
                        loss_val = draw_loss.cpu().data.numpy()
                        avg_loss += loss_val



                        #print('Batch', data.shape, data_tr.shape, 'Loss : ', loss_val)

                        # Calculate the gradients.
                        draw_loss.backward()
                        torch.nn.utils.clip_grad_norm_(draw_model.parameters(), params['clip'])

                        for param, shared_param in zip(draw_model.parameters(), shared_draw_model.parameters()):
                            if shared_param.grad is None: 
                                shared_param._grad = param.grad # sync gradients with shared model

                        shared_draw_optimizer.step()

                        if i != 0 and i%(min(100, len(dataloader_train)-1)) == 0:
                            print('[%d/%d][%d/%d]\tLoss: %.4f'
                                % (epoch+1, params['epoch_num'], i, len(dataloader_train), avg_loss / min(100, len(dataloader_train))))
                            avg_loss = 0
                    #print('[%d/%d][%d/%d]\tLoss: %.4f'
                    #        % (epoch+1, params['epoch_num'], i, len(dataloader_train), avg_loss / len(dataloader_train)))
                #"""
                print('END DRAW TRAINING')
                true_initial_states = []
                rollout = []
                history=[]

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()

        #for param, shared_param in zip(draw_model.parameters(), shared_draw_model.parameters()):
        #    if shared_param.grad is None: shared_draw_model._grad = param.grad # sync gradients with shared model
        #draw_optimizer.step()

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn') # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d
    
    args = get_args()
    args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
    if args.render:  args.processes = 1 ; args.test = True # render mode -> test mode w one process
    if args.test:  args.lr = 0 # don't train in render mode
    args.num_actions = create_env(args.env).action_space.n # get the action space of this game
    print('Num action :', args.num_actions)
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.

    torch.manual_seed(args.seed)
    shared_model = SokobanPolicy(channels=7, memsize=args.hidden, num_actions=args.num_actions).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)

    """ We need to share the Draw model parameter """

    T, channel, W, H =  50, 7, 7, 7 

    # Dictionary storing network parameters.
    params['T'] = T ; params['A'] = H ; params['B'] = W ; params['channel'] = channel
    params['z_size'] = 50 ; params['batch_size'] = 8
    params['read_N'] =  params['write_N'] = H
    params['conv'] = True
    params['epoch_num'] = 10
 
    # Use GPU is available else use CPU.
    device = 'cpu'#torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    params['device'] = device
    

    shared_draw_model = DRAWModel(params).to(params['device']).share_memory()
    draw_optimizer = SharedAdam2(shared_draw_model.parameters(), lr=params['learning_rate'], betas=(params['beta1'], 0.999))

    #draw_optimizer = SharedAdam2(shared_draw_model.parameters(), lr=params['learning_rate'])#, betas=(params['beta1'], 0.999))
    #shared_draw_model.train()
    #checkpoint = False, optim.Adam
    #model_path = 'DRAW/checkpoint/7x7_/{}/model_epoch_30'.format(T)

    #if checkpoint :
    #    shared_draw_model.load_state_dict(torch.load(model_path)['model'])
    #    draw_optimizer.load_state_dict(torch.load(model_path)['optimizer'])    
        #params = torch.load(model_path)['params']

    #train_draw(draw_model, draw_optimizer, params, dataloader, dataloader_test, 'DRAW/checkpoint/7x750')

    #####################################################

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(args.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog(args,'', end='', mode='w') # clear log file
    
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, shared_draw_model, draw_optimizer, params, rank, args, info))
        p.start() ; processes.append(p)
    for p in processes: p.join()
