import requests
import json
import logging
import os
import threading

script_path = os.path.dirname(os.path.realpath(__file__))
telegram_config_path = os.path.join(script_path, "telegram.json")
try:
    with open(telegram_config_path, "r") as fp:
        config = json.load(fp)
except:
    pass


def send_to_bot(message):
    try:
        text = "https://api.telegram.org/bot" \
                + config["api-token"] \
                + "/sendMessage?chat_id=" \
                + config["chat-id"] \
                + "&parse_mode=Markdown&text=" \
                + message
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

