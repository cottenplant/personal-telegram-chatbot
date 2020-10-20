from pathlib import Path


local_env = dict(
    user='',
    chats_raw_json=Path('./data/input/result.json'),
    chats_processed_json=Path(f'./data/output/{user}_chats.json'),
    chats_processed_txt=Path(f'./data/output/{user}_chats.txt'),
    text_sequences=Path(f'./models/{user}_text_sequences.txt'),
    tokenizer_wordcounts=Path(f'./models/{user}_tokenizer_wordcounts.txt'),
    generated_csv=Path(f'./data/output/{user}_chatbot.csv'),
    model_path=Path(f'./models/{user}_chatbot_model.h5'),
    tokenizer_path=Path(f'./models/{user}_chatbot_tokenizer.pickle')
)


class App(env_dict):
  __setters = ["user"]
  __conf = env_dict

  @staticmethod
  def config(name):
    return App.__conf[name]

  @staticmethod
  def set(name, value):
    if name in App.__setters:
      App.__conf[name] = value
    else:
      raise NameError("Name not accepted in set() method")


if __name__ == "__main__":
    App.config("MYSQL_PORT")     # return 3306
    App.set("username", "hi")    # set new username value
    App.config("username")       # return "hi"
    App.set("MYSQL_PORT", "abc") # this raises NameError
