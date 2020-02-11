import requests
import json
import logging
import os
import threading

script_path = os.path.dirname(os.path.realpath(__file__))
telegram_config_path = os.path.join(script_path, "telegram.json")
with open(telegram_config_path, "r") as fp:
    config = json.load(fp)

bot_token = config["api-token"]
bot_chatID = config["chat-id"]

def send_to_bot(message):
    text = "https://api.telegram.org/bot" \
            + bot_token \
            + "/sendMessage?chat_id=" \
            + bot_chatID \
            + "&parse_mode=Markdown&text=" \
            + message

    try:
        response = requests.get(text, timeout=5)
        retval = response.json()
    except:
        retval = None

    return retval


class TelegramHandler(logging.StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        message = self.format(record)
        thread = threading.Thread(target=send_to_bot, args=(message,))
        thread.setDaemon(True)
        thread.start()

