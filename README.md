# COMP-3359-Project-Deep-Q-RL
## A Connect-Four AI that do not win
This GitHub project is built for Nicolas Chan's and Howard Chan's COMP 3359 Group Project use.

The Project Objective is to create a RL agent losing in Connect-Four (7x6).

## Credits
The Connect 4 OpenAI Gym Environment is built upon [gym-connect-four](https://github.com/IASIAI/gym-connect-four) under MIT license.

## File Structure
* './gym_conect_four` is a modified [gym-connect-four](https://github.com/IASIAI/gym-connect-four)
* `./losing_connect_four` contains our code for model training, evaluation, creating deep-Q networks using different components, etc.
* `./saved_models` contains the model architecture (`.h5` files) and weights (`.json` files) of some of our trained models. Corresponding txt files can be used as a quick summary of the model.
* `./Results` contains the saved models, summary text, and graphs of training and evaluations of most of our experimented models.
* `./LICENSES` contains the license of projects we referenced to. Please include all licences in this folder when copying or distributing.
* `README.md` is the document you are reading.
* `LICENSE` is the licence of this project. Please include this licence when copying or distributing.
* `requirements.txt` lists dependencies of this project.
* `setup.py` is the py file needed for setup and installing this project.
* `training.py` is a Python script for training model, and;
* `evaluation.py` is a Python script for evaluating trained model.
* `model_train_and_evaluate.ipynb` is a Jupyter notebook for training and evaluating models. Code in it are mostly copied from `training.py` and `evaluation.py`. Please use this primarily for your trainings and evaluations.
* `playground.ipynb` is a Jupyter notebook for people to play with saved models.

## How to install this project
```bash
git clone -b master "https://github.com/nicolas-chan-42/COMP-3359-Project-Deep-Q-RL.git"  # optional
pip install -e <PATH_TO_THIS_PROJECT_DIR>
```
