## Image Caption Generation using RNNs and LTSMs
### Abstract
In this project, we utilized Recurrent Neural Networks(RNNs) and its ability to capture temporal dynamics to generate descriptive captions for input images in an accurate fashion. In totality, we constructed 3 models: a tuned baseline model using the LSTM module in the decoder, a model using a "vanilla" RNN module, and a model constructed using a different architecture where the input image is passed at each timestep. Caption generation was stochastically or deterministically conducted at test time to generate BLEU-1 and BLEU-4 scores by comparing them to the ground-truth captions. The tuned baseline LSTM model achieved a final test loss of 1.720, a 67.531 BLEU-1 score, and a 8.400 BLEU-4 score. Likewise, the vanilla RNN model achieved a final test loss of 1.921, 64.695 BLEU-1 score, and a 7.410 BLEU-4 score. The Architecture 2 model resulted in a final test loss of 1.722, 68.663 BLEU-1 score, and a 9.264 BLEU-4 score.

### Introduction
Recurrent neural networks are essentially a class of neural networks that allow previous outputs to be used as inputs, whilst allowing for hidden states. This sequential mechanism of operation allows for capturing temporal dynamics over sequential data. In this Programming Assignment, our sequential data in question are the target image captions, which can be interpreted as a sequence of words. Using these ground-truth captions, our objective is to generate a caption that maps to a given image. We approached this problem by breaking up our models into an encoder-decoder structure that communicate with each other. Teacher forcing was conducted during train/validation time to calculate the train/validation loss, and captions were self-generated during testing to compute the BLEU-1 and BLEU-4 scores.

### Methodology
All three models considered were implemented using an encoder-decoder structure. Because CNNs are optimal in producing rich representations of images, we used a Resnet-50 network pre-trained for image classification as our image "encoder" that outputs a fixed-length vector of the image features, by replacing the last hidden layer with a trainable linear layer that will serve as an input to the RNN decoder. For the tuned baseline and the Architecture 2 models, the decoder implements a particular form of a RNN called an LSTM, which encodes knowledge of observed inputs from previous timesteps. This behavior of an LSTM cell is controlled by various gates that can learn what information is relevant or irrelevant from earlier time steps, reducing short-term memory effects. For the vanilla RNN model, the decoder utilizes a traditional RNN cell. These cells take in as input the embeddings of every word in the sentence structure of the ground-truth captions during train time. This is called "teacher-forcing", as the ground-truth captions are used as a teaching signal at every time step (from t = 1), rather than the output sequence of the network from the previous time step. At test time, teacher forcing is turned off, and the encoded features of unseen images in the test set are fed into the decoder at the first time step (t = 0), which the trained model uses to recurrently generate captions at concurrent time steps either stochastically, or deterministically. We deterministically generate captions by taking the maximum of output from the final linear layer, which is a probability distribution representing the vocabulary at each time step. We also generated captions stochastically by sampling from the weighted softmax distribution of the outputs, with a temperature parameter controlling the stochasticity of the sampling. The architecture of the tuned baseline and the vanilla RNN models were the same, in that we used two 300 dimensional embedding layers, 512 units in two hidden layers, followed by a softmax output. In the vanilla RNN model we used the ReLU non-linearity. After testing with different hidden and embedding sizes, we decided that the number of hidden units for the Architecture 2 model should be 768 with embedding size of 600 dimensions. Our hyperparameter search was a strenous one. We noticed that a higher learning rate leads to faster training time but weaker convergence. After testing from learning rates within the range [1e-4, 5e-4], we decided that a learning rate of 1e-4 leads to good convergence while not overly slowing down training time. Convergence was noticed within 8-10 epochs, so we decided to use 10 epochs for every model as the upper limit.

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
