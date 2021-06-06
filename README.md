# CDGnet
Dataset

The data set is generated using MATLAB based on the principle of the Vicsek model, and the simulation system has periodic boundaries. We choose the case where N=4000（and N=300） and ρ=4, and five noises are selected, 0.1, 1.5, 2.5, 3.5, and 5. Our goal was to train the model under five noise points, covering different noise and time steps.

Download dataset

https://figshare.com/articles/dataset/data_rar/13655414
Take N=300 as an example.

Data format

The data is stored in a file in Python's pickle format. Pickle is composed of dictionaries and contains the following entries:
1. positions: initial position
2. angles: initial angle
3. box: the size of the simulated box
4. va: combined speed
5. target_positions: particle positions after each iteration
6. target_angles: the angles after each iteration
7. metadata:A dictionary containing other metadata: 

   noise, noise
   v, particle velocity

Instructions for use
1. Set up the environment, see requirements.txt for details.
2. Set the training parameters in the train_binary.py file, such as data_directory and checkpoint_path.
3. Run the train_binary.py file to get the trained model checkpoint file.
4. Set the training parameters in the train_test.py file.
5. Run the train_test.py file to get the test results.
