# COMP-3359-Project-Deep-Q-RL

This GitHub project is built for Nicolas Chan's and Howard Chan's COMP 3359 Group Project use.

This project includes a Jupyter notebook for training and evaluation named `model_train_and_evaluate.ipynb` and a Jupyter notebook for people to play around with our trained models named `playground.ipynb`. Detailed instructions can be found inside these notebooks. It is suggested that these notebooks should be opened by Google Colab for more convenient operations.

The results of all trained models can be seen in the `\Results` directory. Each folder in this directory contains the saved model (`.h5` and `.json` files), model summary (`.txt`) file and the plots for training and evaluation. This project load saved models from `\saved_models` directory. And new trained models will be automatically saved into this directory for future use.

The Connect 4 OpenAI Gym Environment is built with refrence to gym-connect-four (GitHub: https://github.com/IASIAI/gym-connect-four) under MIT license.  
