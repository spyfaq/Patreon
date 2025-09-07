#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, datetime, requests, re, time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


#  Dropbox  API credentials
DROPBOX_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]

# Patreon  credentials
EMAIL = os.getenv("PATREON_EMAIL")
PASSWORD = os.getenv("PATREON_PASS")

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
def load_file(tier, files):
    # Find TXT file
    txt_file = next(
        (f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.txt")),
        None
    )
    content = None
    if txt_file:
        content = download_dropbox_file(txt_file["path_lower"]).decode("utf-8")

    # Find CSV/XLSX file only for VIP tier
    csv_file = None
    if tier == "VIP":
        csv_file = next(
            (f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.xlsx")),
            None
        )

    return content, csv_file

def html_to_markdown(text: str) -> str:
    """
    Convert simple HTML (like <b>, <i>, <br>) into Patreon-friendly Markdown.
    """
    # Remove anchor tags completely
    text = re.sub(r"<a.*?>.*?</a>", "", text, flags=re.DOTALL)

    # Remove common promo/subscription lines (VIP join, prices, etc.)
    text = re.sub(r"📩.*?\n", "", text)
    text = re.sub(r"💎.*?\n", "", text)

    # Bold -> Markdown
    text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.DOTALL)

    # Italic -> Markdown
    text = re.sub(r"<i>(.*?)</i>", r"*\1*", text, flags=re.DOTALL)

    # Line breaks -> Newline
    text = text.replace("<br>", "\n").replace("<br/>", "\n")

    # Remove extra spaces around bullets
    text = re.sub(r"•\s*", "- ", text)

    # Normalize whitespace
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        return None, None

    title = lines[0]
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    return title, body
    
def post_to_patreon(title, body, IS_VIP=False, file=None):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        # 1. Login
        driver.get("https://www.patreon.com/login")
        time.sleep(5)

        driver.find_element(By.ID, "email").send_keys(EMAIL)
        driver.find_element(By.ID, "password").send_keys(PASSWORD)
        driver.find_element(By.ID, "password").send_keys(Keys.RETURN)
        time.sleep(7)

        # 2. Go to create post
        driver.get("https://www.patreon.com/new-post")
        time.sleep(5)

        # 3. Fill in title
        title_box = driver.find_element(By.CSS_SELECTOR, "textarea[name='title']")
        title_box.send_keys(title)
        time.sleep(1)

        # 4. Fill in body
        body_box = driver.find_element(By.CSS_SELECTOR, "div[data-testid='editor'] div[contenteditable='true']")
        body_box.send_keys(body)
        time.sleep(2)

        # 5. Set audience (Free or Patrons only)
        if IS_VIP:
            #Upload attachment
            upload_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            upload_input.send_keys(file)
            time.sleep(5)  # wait for upload

            audience_btn = driver.find_element(By.XPATH, "//button[contains(., 'Patrons only')]")
            audience_btn.click()
            time.sleep(1)

        # 6. Publish
        publish_btn = driver.find_element(By.XPATH, "//button[contains(., 'Publish now')]")
        publish_btn.click()
        time.sleep(5)

        if IS_VIP:
            print(f"✅ Post created via Selenium for VIP")
        else:
            print(f"✅ Post created via Selenium for Public")

    finally:
        driver.quit()

def main():
    files = list_dropbox_files()
    public,  = load_file('Public', files)
    vip, csv = load_file('VIP', files)

    publictitle, publicbody = html_to_markdown(public)
    viptitle, vipbody = html_to_markdown(vip)

    post_to_patreon(publictitle, publicbody)
    post_to_patreon(viptitle, vipbody, True, csv)

if __name__ == "__main__":
    main()
