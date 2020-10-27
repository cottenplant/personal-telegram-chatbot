___WIP___
# personal-telegram-chatbot

## About
- Telegram makes it simple to export all of your chat history in JSON format
- This project parses that data and filters responses for single user (such as yourself)
- After removing extraneous data such as shared links or photos, messages are fed into a neural network / NLP algorithm
- By generating responses, and a future integration with a Telegram chatbot - you can always be online!
- Avoid pesky conversations without 'ghosting' :D 

## Usage
1. Export your Telegram chats (Linux client works best)
2. Parse the resulting JSON file with the script located at ```chatbot/scripts/parse_transcripts.py```
3. Place parsed JSON in the ```data/input``` directory
4. Train the model (feel free to tune parameters) by running the following command from the root of the directory
```
export PYTHONPATH=$(pwd) && python chatbot/scripts/tokenize_train_fit.py
```
5. Or with Docker
```
docker build -t <namespace>/personal-telegram-chatbot .
docker run -it <namespace>/personal-telegram-chatbot:latest /bin/bash -c "python chatbot/scripts/tokenize_train_fit.py"
docker run <namespace>/personal-telegram-chatbot:latest>
```
6. Read what a tokenized and neural-network digested version of your Telegram self sounds like :D
