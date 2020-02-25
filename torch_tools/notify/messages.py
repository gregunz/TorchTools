import json
import os
from pathlib import Path

import telegram

default_credentials_path = Path(
    os.environ.get('torch_tools_credentials', os.environ.get('HOME'))) / 'notifaime_bot.json'


def send(message, credentials_path=default_credentials_path):
    proxy_url = os.environ.get('http_proxy')
    request_proxy = None
    if proxy_url is not None:
        request_proxy = telegram.utils.request.Request(proxy_url=proxy_url)

    with open(credentials_path, 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
        bot = telegram.Bot(token=token, request=request_proxy)
        bot.sendMessage(chat_id=chat_id, text=message)


if __name__ == '__main__':
    send('test')
