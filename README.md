
![Model](https://github.com/Trevor16gordon/music_generation_rnn/blob/trevor_develop/images/model_overview.jpg)

# Music Generation RNN
This repository is on a project to generate music using a biaxial recurrent neural network. The project is inspired by this [research paper by Daniel J. Johnson](https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9). The code is intended to replicate the orginial model as closely as possible using TensorFlow 2.0. The model uses LSTM layers and a convolution-like reshaping of the input to predict which notes will be played when. 

Checkout the results and explanation [here](https://trevor16gordon.github.io/notes/music_gen_rnn.html)

## Original Paper
Original paper: https://link.springer.com/chapter/10.1007/978-3-319-55750-2_9
Blog: https://www.danieldjohnson.com/2015/08/03/composing-music-with-recurrent-neural-networks/

## This Repo
- main.py: Entry to the most recent best model.
- experiments: Top level entry that combines models with loading specic data, preparing and predicting
- data_preparation: Functions for preparing data for all models
- models: Contains the tensorflow models
- prediction: Contains functions for generation music from trained models
- visualization: Creating plots and saving audio for piano roll
- old_modules: Code for experiments that build up to final model. Not supporting right now.


