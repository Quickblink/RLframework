This repository offers a general framework for Q-learning.

##Features

- different environments can used in the gym format
- recurrent networks and frame-stacking are supported
- multi-processing is used to parallelize data generation and training
- data can be generated according to multiple actor policies (greedy, deterministic, etc.) and even combinations


##Execution

On systems with a configured nvidia-docker installation, the script
`run_main.sh run_id` executes `main.py`, which should be configured to run with the
desired environment and model. The additional command line argument `run_id` to the shell
script or main directly specifies the name suffix for saving results and model
data.