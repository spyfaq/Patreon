#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import requests
import random 

DROPBOX_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_IDS = {
    "Public": os.environ["TELEGRAM_CHAT_ID_TIER1"],
    "VIP": os.environ["TELEGRAM_CHAT_ID_TIER3"],
}

VIP_FALLBACK_MESSAGES = [
    "⚠️ No picks today!\n"
    "Don’t miss out — our <b>VIP members</b> enjoy <b>all daily picks</b> with <i>expert reasoning</i> and a <b>downloadable file</b> every day.\n\n"
    "👉 <a href=\"https://buy.stripe.com/bJe14nfyjfGC5A5abB63K04\">Join VIP now</a> for just €8/month — one winning bet covers it!",

    "❌ No picks today!\n"
    "But inside <b>VIP</b> you’ll find <b>every single prediction</b>, complete with <i>expert insights</i> and a <b>downloadable file</b>.\n\n"
    "🚀 <a href=\"https://buy.stripe.com/bJe14nfyjfGC5A5abB63K04\">Unlock VIP</a> now for only €8/month!",

    "📢 No picks for today.\n"
    "Don’t wait on the sidelines — our <b>VIP community</b> gets <b>all daily bets</b> + detailed <i>reasoning</i> in a handy <b>downloadable file</b>.\n\n"
    "💎 <a href=\"https://buy.stripe.com/bJe14nfyjfGC5A5abB63K04\">Join VIP</a> now and take your betting to the next level!",

    "⚡ No drops today.\n"
    "Meanwhile, <b>VIP members</b> are getting <b>every prediction</b>, backed by <i>AI-powered reasoning</i>, with a <b>downloadable file</b> included daily.\n\n"
    "👉 <a href=\"https://buy.stripe.com/bJe14nfyjfGC5A5abB63K04\">Upgrade to VIP</a> for just €8/month.",

    "ℹ️ No bets today.\n"
    "Remember, <b>BetProphet.AI VIP</b> always delivers <b>all picks</b> with <i>expert reasoning</i> and a <b>downloadable file</b>.\n\n"
    "🔥 <a href=\"https://buy.stripe.com/bJe14nfyjfGC5A5abB63K04\">Join VIP now</a> — one winning slip pays for the month!",

    "⏸️ Free channel is quiet today.\n"
    "But <b>VIP</b> never stops — members receive <b>daily predictions</b>, with <i>detailed reasoning</i> and an easy-to-use <b>downloadable file</b>.\n\n"
    "🚀 <a href=\"https://buy.stripe.com/bJe14nfyjfGC5A5abB63K04\">Go VIP today</a> for just €8/month.",

    "🚫 No picks available right now.\n"
    "Why wait? <b>VIP members</b> have <b>full access</b> to every pick, complete <i>reasoning</i>, and a daily <b>downloadable file</b>.\n\n"
    "👉 <a href=\"https://buy.stripe.com/bJe14nfyjfGC5A5abB63K04\">Join VIP</a> and get the edge today!"
]


# Dropbox folder path
DROPBOX_FOLDER = "/telegram_content"
today_str = datetime.date.today().strftime("%Y-%m-%d")

def list_dropbox_files():
    url = "https://api.dropboxapi.com/2/files/list_folder"
    headers = {"Authorization": f"Bearer {DROPBOX_TOKEN}", "Content-Type": "application/json"}
    payload = {"path": DROPBOX_FOLDER}
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json().get("entries", [])

def download_dropbox_file(path_lower):
    url = "https://content.dropboxapi.com/2/files/download"
    headers = {
        "Authorization": f"Bearer {DROPBOX_TOKEN}",
        "Dropbox-API-Arg": f'{{"path": "{path_lower}"}}'
    }
    r = requests.post(url, headers=headers)
    r.raise_for_status()
    return r.content

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
        send_text_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        text_payload = {"chat_id": CHAT_IDS[tier], "text": content, "parse_mode": "HTML", "disable_web_page_preview": True}
        requests.post(send_text_url, data=text_payload)
        posted_something = True
        print(f"Posted {txt_file['name']} to {tier}")

    if csv_file:
        csv_data = download_dropbox_file(csv_file["path_lower"])
        send_doc_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        files_payload = {"document": (csv_file["name"], csv_data)}
        requests.post(send_doc_url, data={"chat_id": CHAT_IDS[tier]}, files=files_payload)
        print(f"Posted {csv_file['name']} to {tier}")


    # Only send "Hello" to Public if nothing was posted
    if tier == "Public" and not posted_something:
        _message = random.choice(VIP_FALLBACK_MESSAGES)
        send_text_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        text_payload = {"chat_id": CHAT_IDS[tier], "text": _message, "parse_mode": "HTML", "disable_web_page_preview": True}
        requests.post(send_text_url, data=text_payload)
        print(f"Posted to {tier}")
