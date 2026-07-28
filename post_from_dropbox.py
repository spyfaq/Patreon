#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import requests
import random
import date_utils
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

# Fetch and label windows are now aligned (see date_utils.py): a run on
# day X should only ever produce files dated day X. Both today's and
# tomorrow's date are still checked here as a defensive safety net --
# cheap to check, and silently missing a file (as this script used to,
# when it only checked today's date while fetching/labeling disagreed
# about what "today" meant) is worse than one harmless extra lookup that
# finds nothing.
CANDIDATE_DATE_STRS = date_utils.relevant_date_strs()

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
posted_paths = set()  # guards against double-sending if a path somehow matches more than once

for tier in CHAT_IDS.keys():
    for date_str in CANDIDATE_DATE_STRS:
        # Find this date's HTML text file
        txt_file = next(
            (f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{date_str}.txt")),
            None
        )
        # Find this date's xlsx file
        csv_file = next(
            (f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{date_str}.xlsx")),
            None
        )

        if txt_file and txt_file["path_lower"] not in posted_paths:
            content = download_dropbox_file(txt_file["path_lower"]).decode("utf-8")
            if send_telegram_message(BOT_TOKEN, CHAT_IDS[tier], content):
                posted_paths.add(txt_file["path_lower"])
                print(f"Posted {txt_file['name']} to {tier}")
            else:
                print(f"Failed to post {txt_file['name']} to {tier}")

        if csv_file and csv_file["path_lower"] not in posted_paths:
            csv_data = download_dropbox_file(csv_file["path_lower"])
            if send_telegram_document(BOT_TOKEN, CHAT_IDS[tier], csv_file["name"], csv_data):
                posted_paths.add(csv_file["path_lower"])
                print(f"Posted {csv_file['name']} to {tier}")
            else:
                print(f"Failed to post {csv_file['name']} to {tier}")

# Best Bets (value picks from publish_predictions.py) is a VIP-tier bonus
# message, posted separately from the tier loop above because its filename
# starts with neither "Public" nor "VIP" -- the loop's startswith(tier)
# match would never find it. SuggestedBets used to be posted here too; the
# accumulator that produced it has been removed.
for label, prefix in [("Best Bets", "BestBets")]:
    for date_str in CANDIDATE_DATE_STRS:
        bonus_file = next(
            (f for f in files if f["name"].startswith(prefix) and f["name"].endswith(f"{date_str}.txt")),
            None
        )
        if bonus_file and bonus_file["path_lower"] not in posted_paths:
            content = download_dropbox_file(bonus_file["path_lower"]).decode("utf-8")
            if send_telegram_message(BOT_TOKEN, CHAT_IDS["VIP"], content):
                posted_paths.add(bonus_file["path_lower"])
                print(f"Posted {bonus_file['name']} to VIP")
            else:
                print(f"Failed to post {bonus_file['name']} to VIP")
