# Branch Info, Modules and Files

## Info

This branch is created to use Reinforcement Learning to follow the path and complete the task. There are various options for tracking such as:
1. Direct Trajectory Tracking
2. Waypoint Tracking
3. Gate Tracking

where we tested all of the options. You have to choose them indivivually in the wrapper classes using 'None', 'waypoints', and 'gates', resepctively.

This branch does not focus on the trajectory generation, which is something we focused on the other branch called base_path_planning. However, you can find the modules from that branch (old versions) in ./lsy_drone_racing/path_planning directory. This is the merge from 4th July 2024.

You can find trained models and training logs under 
1. ./models/ 
2. ./logs/
respectively.

For training, we created training configurations under ./config/ to spawn the drone in the air. Hence, other .yaml files also include new variables for compatibility.

You can change the model you prefer to use in the simulation in the ./examples/controller.py

For models we prefer different parameters,
1. ppo_lvl1_5sgate_reward.zip: Change the duration between 4.5s-6s. We prefer to have buffer=0.2 or 0.3
2. ppo_lvl2_5_5sgate_reward.zip: Change the duration to 5.8s to 6s. We prefer to have buffer=0.2 or 0.3
3. ppo_lvl1_6s_traj_reward.zip: As we stated, it is unsuccesfull, it is just kept to make our case.

You can find the duration in global_parameters.py and buffer in calc_path_through_gates.py

## Modules
lsy_drone_racing/wrapper.py includes the wrapper environments for training and testing.
lsy_drone_racing/global_parameters.py includes the global training and sim parameters.
lsy_drone_racing/create_waypoints.py includes helper functions for randomly initialized training.
examples/train.py to run the training.
scripts/sm.py to run the sim.

## Files
lvl1*.json files are not important, they are only generated for the presentation.