#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime, requests, os, praw, re


#  Dropbox  API credentials
DROPBOX_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]

#  Reddit API credentials
REDDIT_CLIENT_ID = os.environ["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = os.environ["REDDIT_CLIENT_SECRET"]
REDDIT_USERNAME = "betprophet-AI"
REDDIT_PASSWORD = os.environ["REDDIT_PASSWORD"]
REDDIT_USER_AGENT = "BetProphetAI: v1.0 (by u/BetProphetAI)"


# Target subreddit(s)
SUBREDDITS = ["SoccerBetting", "Sportsbook", "BettingPicks", "sportsbetting", "FixedMatches"]  

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

# Get predictions
def load_file():
    files = list_dropbox_files()
    txt_file = next(
            (f for f in files if f["name"].startswith("VIP") and f["name"].endswith(f"{today_str}.txt")),
            None
        )
    
    content = download_dropbox_file(txt_file["path_lower"]).decode("utf-8")

    return (content)

# transform text
def reddit_fixes(content):
    content = re.sub(r"<b>(.*?)</b>", r"**\1**", content)  # bold
    content = re.sub(r"<i>(.*?)</i>", r"*\1*", content)    # italic

    # Extract only the "Reasoning for Top 5" section
    reasoning_match = re.search(r"Reasoning for Top 5:(.*)", content, re.S)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        # Split by bullet points
        items = re.split(r"•", reasoning_text)
        # Take top 3
        top3 = [item.strip() for item in items[1:] if item.strip()][:3]
        reasoning_text = "\n\n• " + "\n\n• ".join(top3)
    else:
        reasoning_text = "Predictions not found."

    return reasoning_text

# Format Reddit post
def build_post(predictions_text):
    today = datetime.date.today().strftime("%d %B %Y")
    title = f"⚽ BetProphet.AI – 3 Free Football Predictions with Reasoning ({today})"

    body = "Here are today’s top 3 free predictions with reasoning:\n\n"
    body += predictions_text
    body += "\n\n---\n"
    body += "Want more insights? \nFull VIP predictions and detailed analysis are available in the channel 📲!"
    return title, body

def main():
    txt = load_file()
    updated_txt = reddit_fixes(txt)
    title, body = build_post(updated_txt)

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        user_agent=REDDIT_USER_AGENT,
    )

    for sub in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub)
            submission = subreddit.submit(title=title, selftext=body)
            print(f"✅ Posted to r/{sub}: {submission.shortlink}")
        except Exception as e:
            print(f"❌ Error posting to r/{sub}: {e}")

if __name__ == "__main__":
    main()
