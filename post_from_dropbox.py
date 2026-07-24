#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import requests
import random 
#from dotenv import load_dotenv
#load_dotenv() 

DROPBOX_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_IDS = {
    "Public": os.environ["TELEGRAM_CHAT_ID_TIER1"],
    "VIP": os.environ["TELEGRAM_CHAT_ID_TIER3"],
}


# Dropbox folder path
DROPBOX_FOLDER = "/telegram_content"
today_str = datetime.date.today().strftime("%Y-%m-%d")

def list_dropbox_files():
    headers = {
        "Authorization": f"Bearer {DROPBOX_TOKEN}",
        "Content-Type": "application/json"
    }

    entries = []

    # First page
    url = "https://api.dropboxapi.com/2/files/list_folder"
    payload = {
        "path": DROPBOX_FOLDER,
        "recursive": True  # remove if you only want top-level files
    }

    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()

    entries.extend(data.get("entries", []))

    # Remaining pages
    while data.get("has_more"):
        url = "https://api.dropboxapi.com/2/files/list_folder/continue"
        payload = {"cursor": data["cursor"]}

        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        entries.extend(data.get("entries", []))

    return entries

def download_dropbox_file(path_lower):
    url = "https://content.dropboxapi.com/2/files/download"
    headers = {
        "Authorization": f"Bearer {DROPBOX_TOKEN}",
        "Dropbox-API-Arg": f'{{"path": "{path_lower}"}}'
    }
    r = requests.post(url, headers=headers)
    r.raise_for_status()
    return r.content

TELEGRAM_MAX_LEN = 4096  # Telegram's hard limit for sendMessage text


def send_telegram_message(bot_token, chat_id, text):
    """Send a message, splitting it into <=4096-char chunks if needed (Telegram
    rejects longer text outright), and checking the response instead of
    assuming success -- previously a failed send (bad chat_id, over the
    length limit, rate limit, etc.) would still print "Posted" with no
    indication anything went wrong.
    """
    send_text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    chunks = [text[i:i + TELEGRAM_MAX_LEN] for i in range(0, len(text), TELEGRAM_MAX_LEN)] or [text]

    for chunk in chunks:
        payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}
        resp = requests.post(send_text_url, data=payload)
        result = {}
        try:
            result = resp.json()
        except ValueError:
            pass
        if not resp.ok or not result.get("ok", False):
            print(f"❌ Telegram sendMessage failed (status {resp.status_code}): {result.get('description', resp.text)}")
            return False
    return True


def send_telegram_document(bot_token, chat_id, filename, file_bytes):
    send_doc_url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    files_payload = {"document": (filename, file_bytes)}
    resp = requests.post(send_doc_url, data={"chat_id": chat_id}, files=files_payload)
    result = {}
    try:
        result = resp.json()
    except ValueError:
        pass
    if not resp.ok or not result.get("ok", False):
        print(f"❌ Telegram sendDocument failed (status {resp.status_code}): {result.get('description', resp.text)}")
        return False
    return True


files = list_dropbox_files()

for tier in CHAT_IDS.keys():
    # Find today's HTML text file
    txt_file = next(
        (f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.txt")),
        None
    )
    # Find today's CSV file
    csv_file = next(
        (f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.xlsx")),
        None
    )
    posted_something = False

    if txt_file:
        content = download_dropbox_file(txt_file["path_lower"]).decode("utf-8")
        if send_telegram_message(BOT_TOKEN, CHAT_IDS[tier], content):
            posted_something = True
            print(f"Posted {txt_file['name']} to {tier}")
        else:
            print(f"Failed to post {txt_file['name']} to {tier}")

    if csv_file:
        csv_data = download_dropbox_file(csv_file["path_lower"])
        if send_telegram_document(BOT_TOKEN, CHAT_IDS[tier], csv_file["name"], csv_data):
            print(f"Posted {csv_file['name']} to {tier}")
        else:
            print(f"Failed to post {csv_file['name']} to {tier}")
