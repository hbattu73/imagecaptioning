[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7258256&assignment_repo_type=AssignmentRepo)
Change this README to describe how your code works and how to run it before you submit it on gradescope.

## How to Run

### Baseline(Tuned)
1. If you want to change any hyperparameters, go into default_tuned.json and change them
2. Run `python main.py default_tuned` in the main project directory
3. The output of the test will be output in a directory called "experiment_data/default_tuned_experiment"
4. Go into the main directory and open the file "imagedisplay.ipynb". After running this block, you will get an output of the 3 best images and 3 worst images from your experiment. If you have multiple experiments in the "experiment_data" directory, you will have to look through the output to find the file name for which experiment you ran.
5. If you want to run different experiments on the same model, you'll first have to rename the .pkl file in the "experiment_data/default_tuned_experiment" directory so it doesn't get overridden. Then, comment out the line `exp.run()` in main.py and change the .json file to whatever parameters you'd like to test and repeat step 4 to view.


### Vanilla RNN
1. If you want to change any hyperparameters, go into vanilla.json and change them
2. Run `python main.py vanilla` in the main project directory
3. The output of the test will be output in a directory called "experiment_data/vanilla_experiment"
4. Repeat steps 4 and 5 from the baseline experiment

### Architecture 2
1. If you want to change any hyperparameters, go into arch2.json and change them
2. Run `python main.py arch2` in the main proj
3. The output of the test will be output in a directory called "experiment_data/arch2_experiment"
4. Repeat steps 4 and 5 from the baseline experiment
