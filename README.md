# woulda

[*Woulda, Coulda, Shoulda : Counterfactually-Guided Policy Search*](https://arxiv.org/abs/1811.06272), **l. Buesing, T. Weber, Y. Zwols et al.**, 2018

What *would have happened*, had you chose a different action ? Can we simulate counterfactual (and plausible) experiences instead of large amounts of real experiences (which can be costly to acquire)? The authors proposed a new procedure for policy evaluation using counterfactual experiences (**CF-PE**) as well as a new algorithm for learning policies in PO-MDPs from off-policy experience, called the Counterfactually-Guided Policy Search (**CF-GPS**). It explicitely considers alternative outcomes, allowing the algorithm to make better use of experience data.

This repo is an attempt to replicate the algorithms and the results of the paper.

#### The Sokoban Game

See notebook [*Woulda Coulda Shoulda - the Sokoban Game*](https://nbviewer.jupyter.org/github/dam-grassman/woulda/blob/master/Woudla%20Coulda%20Shoulda%20-%20The%20Sokoban%20Game.ipynb) for more information.

#### Learn a policy using A3C

Instead of IMPALA (mentionned several times in the paper), we used an implemenatation of A3C called [baby_a3c](https://github.com/greydanus/baby-a3c). 

```cmd

python baby_a3c --env Sokoban-small-v0 --processes 8
```

#### DRAW 

In order to generate initial states, the authors used a generative algo called [DRAW](https://arxiv.org/pdf/1502.04623.pdf) whose implementation has been inspired from this [repo](https://github.com/Natsu6767/Generating-Devanagari-Using-DRAW).

For more information on how DRAW works and is applied in our context, refer to the notebook [*Woulda Coulda Shoulda - DRAW*](https://nbviewer.jupyter.org/github/dam-grassman/woulda/blob/master/Woudla%20Coulda%20Shoulda%20-%20Draw.ipynb)

#### Step-by-Step

A more in-depth analysis of the paper can be found in the notebook [*Woulda Coulda Shoulda - Step by Step*](https://nbviewer.jupyter.org/github/dam-grassman/woulda/blob/master/Woudla%20Coulda%20Shoulda%20-%20Step%20by%20Step.ipynb) 
