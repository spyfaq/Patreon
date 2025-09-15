#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, datetime, requests, re, warnings, asyncio, time
from playwright.async_api import async_playwright

warnings.filterwarnings('ignore')

# Dropbox API credentials
DROPBOX_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]

# Patreon credentials
EMAIL = os.getenv("PATREON_EMAIL")
PASSWORD = os.getenv("PATREON_PASS")

# Dropbox folder path
DROPBOX_FOLDER = "/telegram_content"
today_str = datetime.date.today().strftime("%Y-%m-%d")

def list_dropbox_files():
    print('Collecting Dropbox files..')
    url = "https://api.dropboxapi.com/2/files/list_folder"
    headers = {"Authorization": f"Bearer {DROPBOX_TOKEN}", "Content-Type": "application/json"}
    payload = {"path": DROPBOX_FOLDER}
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json().get("entries", [])

def download_dropbox_file(path_lower):
    print('Downloading Dropbox files..')
    url = "https://content.dropboxapi.com/2/files/download"
    headers = {
        "Authorization": f"Bearer {DROPBOX_TOKEN}",
        "Dropbox-API-Arg": f'{{"path": "{path_lower}"}}'
    }
    r = requests.post(url, headers=headers)
    r.raise_for_status()
    return r.content

def load_file(tier, files):
    print(f'Loading {today_str} files..')
    txt_file = next((f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.txt")), None)
    content = None
    if txt_file:
        content = download_dropbox_file(txt_file["path_lower"]).decode("utf-8")

    csv_file = None
    if tier == "VIP":
        csv_file = next((f for f in files if f["name"].startswith(tier) and f["name"].endswith(f"{today_str}.xlsx")), None)

    return content, csv_file

def html_to_markdown(text: str):
    print(f'Converting text to markdown..')
    text = re.sub(r"<a.*?>.*?</a>", "", text, flags=re.DOTALL)
    text = re.sub(r"📩.*?\n", "", text)
    text = re.sub(r"💎Just €8.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<b>(.*?)</b>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<i>(.*?)</i>", r"\1", text, flags=re.DOTALL)
    text = text.replace("<br>", "\n").replace("<br/>", "\n")

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None, None
    return lines[0], "\n".join(lines[1:]) if len(lines) > 1 else ""

async def post_to_patreon(title, body, IS_VIP=False, file=None):
    print(f'Launching Playwright..')
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-features=VizDisplayCompositor",
                "--disable-blink-features=AutomationControlled",
            ]
        )
        context = await browser.new_context()
        page = await context.new_page()

        print("Login to Patreon..")
        await page.goto("https://www.patreon.com/login", timeout=60000)

        # Login
        await page.wait_for_selector("input[type='email']", timeout=30000)
        await page.fill("input[type='email']", EMAIL)
        await page.keyboard.press("Enter")

        await page.wait_for_selector("input[type='password']", timeout=30000)
        await page.fill("input[type='password']", PASSWORD)
        await page.keyboard.press("Enter")

        await page.wait_for_timeout(5000)

        # Navigate to new post page
        
        print(f'Redirect to post page..')
        await page.goto("https://www.patreon.com/posts/new?postType=text_only", timeout=60000)

        time.sleep(10)
        await page.mouse.click(1, 1)
        # Audience
        print(f'Setting audience..')
        if IS_VIP:
            if file:
                print(f'Uploading VIP file..')
                upload_input = page.locator("#add-attachments-button input[type='file']")
                await upload_input.set_input_files(file["path_lower"])
                await page.wait_for_timeout(5000)
                radio_btn = page.locator("//input[@type='radio' and @value='paid']")
                await radio_btn.click()
        else:
            radio_btn = page.locator("//input[@type='radio' and @value='public']")
            await radio_btn.click()

        # Fill post 
        print(f'Writing post..')
        body = body.replace("Reasoning for Top 5:", " \n 💡 Reasoning for Predictions:")
        await page.fill("div.ProseMirror.remirror-editor", body)
        await page.fill("textarea[aria-label='Title']", title)

        # Publish
        print(f'Publishing..')
        await page.click('button[data-tag="make-a-post-action-publish"]')
        await page.wait_for_timeout(3000)

        print("✅ Post created via Playwright", "VIP" if IS_VIP else "Public")

        await browser.close()

async def main():
    files = list_dropbox_files()
    public, txt = load_file('Public', files)
    vip, csv = load_file('VIP', files)

    publictitle, publicbody = html_to_markdown(public)
    viptitle, vipbody = html_to_markdown(vip)

    await post_to_patreon(publictitle, publicbody)
    await post_to_patreon(viptitle, vipbody, True, csv)

if __name__ == "__main__":
    asyncio.run(main())
