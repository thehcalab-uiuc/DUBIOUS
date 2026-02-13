#  From Legible to Inscrutable Trajectories: (Il)legible Motion Planning Accounting for Multiple Observers 
## Abstract

In cooperative environments, such as in factories or assistive scenarios, it is important for a robot to communicate its intentions to observers, who could be either other humans or robots. A legible trajectory allows an observer to quickly and accurately predict an agent's intention. In adversarial environments, such as in military operations or games, it is important for a robot to not communicate its intentions to observers. An illegible trajectory leads an observer to incorrectly predict the agent's intention or delays when an observer is able to make a correct prediction about the agent's intention. However, in some environments there are multiple observers, each of whom may be able to see only part of the environment, and each of whom may have different motives. In this work, we introduce the Mixed-Motive Limited-Observability Legible Motion Planning (MMLO-LMP) problem, which requires a motion planner to generate a trajectory that is legible to observers with positive motives and illegible to observers with negative motives while also considering the visibility limitations of each observer. We highlight multiple strategies an agent can take while still achieving the problem objective. We also present DUBIOUS, a trajectory optimizer that solves MMLO-LMP. Our results show that DUBIOUS can generate trajectories that balance legibility with the motives and limited visibility regions of the observers. Future work includes many variations of MMLO-LMP, including moving observers and observer teaming. 

[See full work on arXiv](https://arxiv.org/abs/2602.09227)

## Running the Code
This repository requires the following:

```
matplotlib==3.10.8
numpy==2.3.5
rustworkx==0.17.1
scipy==1.16.3
shapely==2.1.2
tqdm==4.67.1
```

Experiment configurations can be found in `experiment_configs`. 
A single experiment can be run with 
```
python3 main_experiment.py --config experiment_configs/{experiment_filename}.json
```
