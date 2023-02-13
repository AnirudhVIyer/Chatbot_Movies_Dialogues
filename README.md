# README.md

# About

- A Seq2Seq, attention based chatbot trained on the Cornell Movie Corpus

# Setting up the environment

python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .

# Model options

- Config the congig/config.py according to the desired model
- change zip_location according to the local structure

# Extract data

- run main.extract_zip()
- run main.load_data()

# Train a model

- According to the required configs, train a model
- run the main.train()

# Interact with Chatbot

- Select the model you want to load, change the congif variables
- Trained models are stored in data/save
- Models are loaded automatically based on the config variables
- Run main.predict()
- Interact with the chatbot as you wish!


#About data folder
-Training according to this should locally create a Data folder
-Data Folder will contain checkpoints for the trained model
-Once trained, can predict
- data folder too large to commit 
