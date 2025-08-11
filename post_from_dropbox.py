import os
import datetime
import requests

DROPBOX_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_IDS = {
    "tier1": os.environ["TELEGRAM_CHAT_ID_TIER1"],
    "tier2": os.environ["TELEGRAM_CHAT_ID_TIER2"],
    "tier3": os.environ["TELEGRAM_CHAT_ID_TIER3"],
}

# Dropbox folder path
DROPBOX_FOLDER = "/telegram_content"
today_str = datetime.date.today().strftime("%Y-%d-%m")

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
    md_file = next((f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.md")), None)
    csv_file = next((f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.csv")), None)

    if md_file:
        content = download_dropbox_file(md_file["path_lower"]).decode("utf-8")
        send_text_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        text_payload = {"chat_id": CHAT_IDS[tier], "text": content, "parse_mode": "Markdown"}
        requests.post(send_text_url, data=text_payload)
        print(f"Posted {md_file['name']} to {tier}")

    if csv_file:
        csv_data = download_dropbox_file(csv_file["path_lower"])
        send_doc_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        files_payload = {"document": (csv_file["name"], csv_data)}
        requests.post(send_doc_url, data={"chat_id": CHAT_IDS[tier]}, files=files_payload)
        print(f"Posted {csv_file['name']} to {tier}")