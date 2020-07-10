import requests
import json
import logging
import os
import threading

script_path = os.path.dirname(os.path.realpath(__file__))
telegram_config_path = os.path.join(script_path, "telegram.json")
config = dict()
try:
    with open(telegram_config_path, "r") as fp:
        config = json.load(fp)
except:
    if "TELEGRAM_BOT_TOKEN" in os.environ and "TELEGRAM_BOT_CHAT_ID" in os.environ:
        config = {
                "api-token": os.environ["TELEGRAM_BOT_TOKEN"],
                "chat-id": os.environ["TELEGRAM_BOT_CHAT_ID"],
        }
        with open(telegram_config_path, "w") as fp:
            json.dump(config, fp, indent=4)


def send_to_bot(message, token, chatid):
    try:
        text = "https://api.telegram.org/bot" \
                + token \
                + "/sendMessage?chat_id=" \
                + chatid \
                + "&text=" \
                + message
        response = requests.get(text, timeout=5)
        retval = response.json()
    except:
        retval = None

    return retval


class TelegramHandler(logging.StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = config.get("api-token", None)
        self.chatid = config.get("chat-id", None)

        if self.token is not None and self.chatid is not None:
            self.useable = True
        else:
            self.useable = False

    def emit(self, record):
        if self.useable:
            message = self.format(record)

            thread = threading.Thread(
                    target=send_to_bot,
                    args=(message, self.token, self.chatid))

            thread.setDaemon(True)
            thread.start()

