#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, datetime, requests, re, time, warnings, pyperclip, tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

warnings.filterwarnings('ignore')

#from dotenv import load_dotenv
#load_dotenv()

#  Dropbox  API credentials
DROPBOX_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]

# Patreon  credentials
EMAIL = os.getenv("PATREON_EMAIL")
PASSWORD = os.getenv("PATREON_PASS")

# Dropbox folder path
DROPBOX_FOLDER = "/telegram_content"

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

# Get predictions
def load_file(tier, files):
    print(f'Loading {today_str} files..')
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
    print(f'Converting text to markdown..')
    # Remove anchor tags completely
    text = re.sub(r"<a.*?>.*?</a>", "", text, flags=re.DOTALL)

    # Remove common promo/subscription lines (VIP join, prices, etc.)
    text = re.sub(r"📩.*?\n", "", text)
    text = re.sub(r"💎Just €8.*", "", text, flags=re.DOTALL)

    # Remove bold/italic tags completely
    text = re.sub(r"<b>(.*?)</b>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<i>(.*?)</i>", r"\1", text, flags=re.DOTALL)

    # Line breaks -> Newline
    text = text.replace("<br>", "\n").replace("<br/>", "\n")

    # Normalize whitespace
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if not lines:
        return None, None

    title = lines[0]
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    return title, body

def handle_cookie_banner(driver):
    print(f'Closing cookies..')
    try:
        # Try clicking Reject non-essential
        time.sleep(5)
        reject_btn = driver.find_element(By.XPATH, "//button[.//span[text()='Reject non-essential']]")
        reject_btn.click()
        print("Rejected cookies")
    except:
        # If reject button not found, remove banner
        driver.execute_script("""
            const el = document.getElementById("transcend-consent-manager");
            if (el) el.remove();
        """)
        print("Removed cookie banner manually")

def close_optional_dialog(driver):
    print(f'Closing popup dialog..')
    try:
        time.sleep(5)
        close_btn =  driver.find_element(By.XPATH, "//button[@aria-label='Close Dialog']")
        close_btn.click()
        print("Dialog closed")
    except Exception as e:
        print(str(e))

def select_create_post_option(driver, option_name="Post"):
    print(f'Creating post..')
    try:
        post = driver.find_element(By.XPATH, "//input[@aria-label='Create post']")
        post.click()

        # Wait for the dropdown/modal to appear
        post = driver.find_element(By.XPATH, "//p[text()='Post']")
        post.click()
        print(f"✅ '{option_name}' selected")
    except Exception as e:
        print(f"❌ Failed to select '{option_name}': {e}")

def post_dialog(driver):
    print(f'Closing post dialog..')
    try:
        # Wait for the close button to be clickable
        time.sleep(5)
        close_button = driver.find_element(By.CSS_SELECTOR, 'button[data-tag="dialog-close-icon"]')
        close_button.click()
        print("Dialog closed successfully")
    except Exception as e:
        print(f"Could not close dialog: {e}")

def type_in_prosemirror(driver, text, editor_selector="div.ProseMirror"):
    """
    Focus and type text into a ProseMirror editor.
    
    :param driver: Selenium WebDriver instance
    :param text: Text to type
    :param editor_selector: CSS selector for ProseMirror container (default = 'div.ProseMirror')
    """
    print(f'Writting body..')
    # Locate the editor
    editor = driver.find_element(By.CSS_SELECTOR, editor_selector)

    # Ensure editor is in view
    driver.execute_script("arguments[0].scrollIntoView(true);", editor)
    time.sleep(0.5)

    # Focus and click inside editor
    ActionChains(driver).move_to_element(editor).click().perform()
    time.sleep(0.2)

    # Send keys
    ActionChains(driver).send_keys(text).perform()

def post_to_patreon(title, body, IS_VIP=False, file=None):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    temp_profile = tempfile.mkdtemp()
    chrome_options.add_argument(f"--user-data-dir={temp_profile}")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    #driver = webdriver.Chrome(executable_path="misc\chromedriver.exe", options=chrome_options)
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f'Login to Patreon..')
        # Login
        driver.get("https://www.patreon.com/login")
        time.sleep(5)

        handle_cookie_banner(driver)
        time.sleep(10)
        email_input = driver.find_element(By.XPATH, "//input[@aria-label='Email']")

        email_input.send_keys(EMAIL)
        email_input.send_keys(Keys.RETURN)
        time.sleep(10)
        password_input = driver.find_element(By.XPATH, "//input[@aria-label='Password']")
        time.sleep(5)
        password_input.send_keys(PASSWORD)
        password_input.send_keys(Keys.RETURN)

        time.sleep(5)

        # Go to create post
        print(f'Redirect to post page..')
        driver.get("https://www.patreon.com/posts/new?postType=text_only")
        time.sleep(5)
        handle_cookie_banner(driver)
        close_optional_dialog(driver)
        time.sleep(5)
       

        # Post Content
        print(f'Writting post..')
        body = body.replace("Reasoning for Top 5:", " \n 💡 Reasoning for Predictions:")
        lines = body.split("\n") 
        time.sleep(20)      
        body_editor = driver.find_element(By.CSS_SELECTOR, "div.ProseMirror.remirror-editor")
        js_append_lines = """
            let editor = arguments[0];
            let lines = arguments[1];

            editor.innerHTML = "";  // Clear existing content

            for (let i = 0; i < lines.length; i++) {
                let lineNode = document.createTextNode(lines[i]);
                editor.appendChild(lineNode);
                if (i < lines.length - 1) {
                    let br = document.createElement("br");
                    editor.appendChild(br);
                }
            }
            editor.dispatchEvent(new Event('input', { bubbles: true }));
            """
        driver.execute_script(js_append_lines, body_editor, lines)
        time.sleep(1)


        # Set audience (Free or Patrons only)
        print(f'Setting audience..')
        if IS_VIP:
            #Upload attachment
            print(f'Uploading VIP file..')
            upload_input = driver.find_element(By.CSS_SELECTOR, "#add-attachments-button input[type='file']")
            upload_input.send_keys(file["path_lower"])
            time.sleep(5)  # wait for upload

            audience_btn = driver.find_element(By.XPATH, "//button[contains(., 'Patrons only')]")
            audience_btn.click()
            time.sleep(1)
        else:
            radio_btn = driver.find_element(By.XPATH, "//input[@type='radio' and @value='public']")
            radio_btn.click()

        # Need to write title at the end so react doesnt clear it
        title_box = driver.find_element(By.XPATH, "//textarea[@aria-label='Title']")
        pyperclip.copy(title)

        title_box.click()
        title_box.clear()
        time.sleep(0.2)

        title_box.send_keys(Keys.CONTROL, 'v')

        # 6. Publish
        print(f'Publishing..')
        publish_btn = driver.find_element(By.XPATH, '//button[@data-tag="make-a-post-action-publish"]')
        publish_btn.click()
        time.sleep(3)

        if IS_VIP:
            print(f"✅ Post created via Selenium for VIP")
        else:
            print(f"✅ Post created via Selenium for Public")

    finally:
        driver.quit()

def main():
    files = list_dropbox_files()
    public, txt = load_file('Public', files)
    vip, csv = load_file('VIP', files)

    publictitle, publicbody = html_to_markdown(public)
    viptitle, vipbody = html_to_markdown(vip)

    post_to_patreon(publictitle, publicbody)
    post_to_patreon(viptitle, vipbody, True, csv)

if __name__ == "__main__":
    main()
