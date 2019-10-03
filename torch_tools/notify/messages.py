import json
import os

import telegram


def send(message, credentials_path='../credentials/notifaime_bot.json'):
    proxy_url = os.environ.get('https_proxy', None)
    request_proxy = None if proxy_url is None else telegram.utils.request.Request(proxy_url=proxy_url)
    with open(credentials_path, 'r') as keys_file:
        k = json.load(keys_file)
        token = k['telegram_token']
        chat_id = k['telegram_chat_id']
        bot = telegram.Bot(token=token, request=request_proxy)
        bot.sendMessage(chat_id=chat_id, text=message)
