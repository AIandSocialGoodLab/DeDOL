## Implementation for the DeDOL algorithm proposed in 'Deep Reinforcement Learning for Green Security Games with Real-Time Information', AAAI 2019.

For more details of the algorithm, please refer to the paper [Deep Reinforcement Learning for Green Security Games with Real-Time Information](https://arxiv.org/abs/1811.02483) 

### Pre-requsite

- Tensorflow GPU     
- cvxopt
- nashpy

### Basic Description

- env.py: the GSG-I game model class
- DeDOL.py: the main file for running the DeDOL algorithms
- DeDOL_util.py: helper functions for DO.py
- DeDOL_Global_Retrain.py: for loading the models trained in local modes, and then run more iterations in gloabl mode training
- GUI_util.py: helper functions for showing the game using GUI
- GUI.py: test the performance of trained DQNs using GUI.
- maps.py: helper functions for generate different kinds of maps
- patroller_cnn.py: the patroller CNN strategy representation
- poacher_cnn.py: the poacher CNN strategy representation
- patroller_rule.py: our designed heuristic parameterized random walk patroller
- poacher_rule.py:  our designed heuristic parameterized random walk patroller
- patroller_randomsweeping.py: our desinged heuristic random sweeping patroller
- replay_buffer.py: the replay buffer data structure needed for DQN training and prioterize experience replay
- AC_patroller: the actor_critic patroller. Performs poor, not adopted in the DeDOL algorithm.

**Most of the files include further detailed comments**

### How to run the DeDOL algorithm?

- First run **DeDOL.py** for different local modes or pure global mode.
  - The default training parameters should work well. You can also explore by yourself. 
  - To run in different local modes, change the 'po_location' parameter from 0 to 3, representing four different entering points. The code will automatically generate new directors saving DQN models trained in different local modes, for later loading in the DeDOL_Global_Retrain.py file.
  - E.g. the command 'python DeDOL.py --row_num 5 --po_location 0 --map_type gauss' will run the DeDOL algorithm in a 5x5 grid, Mixture Gaussian Map, and the poacher will always enter the grid world from the left-top corner. The trained DQNs will be stored in the direct './Results_55_gauss_mode0/'.
  - The training of DQNs could really be time-consuming in the convoluted GSG-I game. And several iterations of DeDOL would be requried to evolve a resonalbe strategy profile. Be patient :).
  
- To collect the DQNs and run more DO iterations in global mode:

  - You should first run **DeDOL.py**  in all local modes.
  - Run **DeDOL_Global_retrain.py**. Set the **load_path** parameter to be compatible with the **save_path** parameter you used in DeDOL.py to load the previous DQNs trained in local modes. The **save_path** parameter should omit the last number that specifying the mode, as it will auto collect all DQNs trained in all local loads. E.g if **save_path** is **./Results_33_random_mode0/** to  **./Results_33_random_mode3/** , the  **load_path** should be  **./Results_33_random_mode**. 

- To visualize the game process:
  - run GUI.py with arg 'load' set False will visualize the behaviour of a parameterized poacher and a random sweeping patroller. You can change parameters like 'row_num', 'map_type', 'max_time' for fun.
  - If you want to visualize the performance of trained DQNs, run GUI.py with arg 'load' set be True, and set the corresponding  'pa_load_path' and 'po_load_path' args to the path where you stored your DQN models.
  - A pretrained patroller DQN against a heuristic parameterized poacher, and a pretrained poacher DQN against a randomsweeping patroller (in 7x7 grid world) is contained in the *Pre-trained_Models* diretory. 

   



