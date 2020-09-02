from baby_a3c import create_env, SokobanPolicy, prepro, F
import torch
import time
import numpy as np


def run_test(model, env, render, sleep_time = 0.1, stochasticity = 1, **kwargs):
    model.eval()
    #if reset_state is  None :
    #    state = torch.tensor(prepro(env.reset())) # get first state
    #else :
    #    state = torch.tensor(prepro(env.reset(rand=reset_state))) 
    state = torch.tensor(prepro(env.reset(**kwargs))) 
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    hx = torch.zeros(1, 256) if done else hx.detach()  # rnn activation vector
    values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

    done = False
    while not done:
        episode_length += 1
        value, logit, hx = model((state.view(1,7,7,7), hx))
        logp = F.log_softmax(logit, dim=-1)

        if np.random.random() > stochasticity :
            action = np.random.randint(env.action_space.n)
            state, reward, done, _ = env.step(action)
        else :
            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            state, reward, done, _ = env.step(action.numpy()[0])
        
        if render: 
            time.sleep(sleep_time)
            env.render()

        state = torch.tensor(prepro(state)) ; epr += reward
        done = done or episode_length >= 50 # don't playing one ep for too long

        info['frames'].add_(1) ; num_frames = int(info['frames'].item())
        
        if done: # update shared data
            info['episodes'] += 1
            interp = 1 if info['episodes'][0] == 1 else 1 - 0.99
            info['run_epr'].mul_(1-interp).add_(interp * epr)
            info['run_loss'].mul_(1-interp).add_(interp * eploss)

        if done: # maybe print info.
            episode_length, epr, eploss = 0, 0, 0
            #state = torch.tensor(prepro(env.reset()))  
        
    if render:
        env.render()
        time.sleep(5)
        env.close()

    return info 
  
        
