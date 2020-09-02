import numpy as np
import torch

def fix_player(arr):
    mininum, maximum = np.min(arr[:,:,:]), np.max(arr[:,:,:])
    ind = np.unravel_index(np.argmax(arr[5, :,:], axis=None), arr[5, :,:].shape)
    arr[5, :, :] = mininum
    arr[6, :, :] = mininum
    arr[3, :, :] = mininum
    arr[5, ind[0], ind[1]] = np.max(arr[:,:,:])+1
    return arr

def fix_item(arr, item):
    mininum, maximum = np.min(arr[:,:,:]), np.max(arr[:,:,:])
    indices = np.argwhere(arr.argmax(0)==item)
    if len(indices) == 2:
        pass
    elif len(indices) > 2:
        l = []
        for ind in indices:
            l.append((ind, arr[item, ind[0], ind[1]]))
        l = sorted(l, key = lambda tup : tup[1])
        arr[item, :,:] = mininum
        for ind, val  in l[-2:]:
            arr[item, ind[0],ind[1]] = maximum
    elif len(indices) == 1:
        ind = indices[0]
        arr_copy = arr[item, :, :].copy()
        arr[item, :,:] = mininum
        arr[item,ind[0], ind[1]] = maximum
        arr_copy[ind[0], ind[1]] = mininum 
        ind = np.unravel_index(np.argmax(arr_copy, axis=None), arr_copy.shape)
        arr[item, ind[0],ind[1]] = maximum
    else :
        arr_copy = arr[item, :, :].copy()
        arr[item, :,:] = mininum
        ind = np.unravel_index(np.argmax(arr_copy, axis=None), arr_copy.shape)
        arr[item, ind[0],ind[1]] = maximum
        arr_copy[ind[0], ind[1]] = mininum 
        ind = np.unravel_index(np.argmax(arr_copy, axis=None), arr_copy.shape)
        arr[item, ind[0],ind[1]] = maximum
    return arr

def predict_initial_state(draw_model, history, transpose = False, fixed = False, device = 'cpu') :
    draw_model.eval()
    with torch.no_grad():
        draw_model.cs = [0] * draw_model.Z
        if transpose :
            history = np.transpose(history.reshape(-1,7,7,7), (0,3,1,2))
        else :
            history = history.reshape(-1,7,7,7)
        draw_model.forward(torch.tensor(history.reshape(-1,7,7,7), dtype = torch.float32).unsqueeze(0).to(device))
        arr = draw_model.cs[-1].reshape(7,7,7).cpu().data.numpy()#, (1,2,0))
        if fixed :
            arr = fix_player(arr)
            arr = fix_item(arr,2)
            arr = fix_item(arr,4)
        return arr

def quality_check(initial_state):
    nb_agents = np.where(initial_state==5)
    nb_boxes = np.where(initial_state==4)
    nb_targets = np.where(initial_state==2)
    return len(nb_agents[0]), len(nb_boxes[0]), len(nb_targets[0])

def start_allowed(initial_state):
    ag, _ , _= quality_check(initial_state)
    if ag == 0 or ag > 1 : 
        return False
    else :
        return True
def reward_correction(initial_state, reward):
    ag, bx, tg = quality_check(initial_state)
    if tg == 1 : 
        return min(1, reward-1)
    elif tg < 1 :
        return 0
    elif tg > 2 :
        return reward+1
    elif ag == 0 :
        return 0
    elif ag > 1 : 
        return 0
    else :
        return reward
