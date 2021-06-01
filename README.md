This repository offers a general framework for Q-learning.

# Core Features

- multi-processing is used to parallelize data generation and training
- different environments can used in the gym format
- recurrent networks and frame-stacking are supported
- data can be generated according to multiple actor policies (greedy, deterministic, etc.) and even combinations

# Execution

On systems with a configured nvidia-docker installation, the script
`run_main.sh run_id` executes `main.py`, which should be configured to run with the
desired environment and model. The additional command line argument `run_id` to the shell
script or main directly specifies the name suffix for saving results and model
data.

`PlotResults.ipynb` can be used to create graphs as seen below.

# Internal Structure

The framework uses two separate processes (blue and red in the graphic) for data generation and training.
Additionally, gyms vector environments with internal multiprocessing are employed. The two processes communicate
via shared memory (dotted area). The two processes are synced in regular intervals. The buffer in shared memory is
then copied to the replay buffer used for training and the training network is copied to the data generation
process in exchange. If GPU memory is sufficient for storing the replay buffer, the training process does not rely
on any data from CPU between synchronisations. Additionally, the training does not have to wait for the
environment.


![Code Structure](imgs/code_structure.svg)

#Example: Breakout

Using a standard convolutional architecture (details in `main.py`) with variations in frame stacking and
the last layer (FeedForward or LSTM), a capable agent can be trained in a few hours of wall-clock time.
One example episode can be seen here:

![An AI agent playing Breakout](imgs/some_game.gif)


The comparison of different methods below shows that frame stacking that is usually applied for Atari games
can be replaced by using an LSTM. Combining both techniques does not provide an additional advantage.

![Breakout Results](imgs/results.svg)

