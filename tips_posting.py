#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import selenium_func as sel
import datetime, os, pyperclip, pyautogui, time, random
from jsonlogger_class import JSONLogger
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import bettexts

LOGPATH = 'logs/posting/'
LOGNAME = '{date}_tipspost_logs'
PUBLISHPATH = 'publish/'

def _inita():
    global USERMAIL, USERPASS, TIER1TITLE, TIER1TEXT, TIER2TITLE, TIER2TEXT, TIER3TITLE, TIER3TEXT
    with open('init.cfg', 'r', encoding='utf-8') as FIL:
        data = FIL.readlines()
        FIL.close()
    
    USERMAIL = data[0].split(' = ')[-1].strip()
    USERPASS = data[1].split(' = ')[-1].strip()

    random_tier_1_post = random.choice(bettexts.tier_1_posts)
    random_tier_2_post = random.choice(bettexts.tier_2_posts)
    random_tier_3_post = random.choice(bettexts.tier_3_posts)
    TIER1TITLE = random_tier_1_post['Title']
    TIER1TEXT = random_tier_1_post['Text']
    TIER2TITLE = random_tier_2_post['Title']
    TIER2TEXT = random_tier_2_post['Text']
    TIER3TITLE = random_tier_3_post['Title']
    TIER3TEXT = random_tier_3_post['Text']
    return

def login_n_post(Titletext, Textdata, Tier):
    logger.log('info', 'Process started..')
    server, driver = sel.start_server_and_driver()
    complete_url = 'https://www.patreon.com/login'
    driver.get(complete_url)

    # email
    email = driver.find_element(By.NAME, 'email')
    [email.send_keys(c) for c in USERMAIL]

    # Continue
    temp = driver.find_element(By.XPATH, "//button[. = 'Continue']")
    temp.click()

    # current-password
    email = driver.find_element(By.NAME, 'current-password')
    [email.send_keys(c) for c in USERPASS]

    # Continue
    temp = driver.find_element(By.XPATH, "//button[. = 'Continue']")
    temp.click()
    logger.log('info', 'Logged in..')
    time.sleep(5)
    cookies = driver.get_cookies()

    for cookie in cookies:
        driver.add_cookie(cookie)
    driver.get('https://www.patreon.com/user')
    logger.log('info', 'Creator page loaded..')

    # close pop up
    try:
        temp = driver.find_element(By.XPATH, "//button[@aria-label='Close Dialog']")
        logger.log('info', 'Dialog closed..')
        temp.click()
    except:
        pass

    try:
        temp = driver.find_element(By.XPATH, "//button[@aria-label='Close']")
        logger.log('info', 'Dialog2 closed..')
        temp.click()
    except:
        pass

    # Create - aria-label="Create post" or 
    temp = driver.find_element(By.XPATH, '//button[@aria-label="Create post"]')
    temp.click()

    # Create Post
    temp = driver.find_element(By.XPATH, '/html/body/div[2]/div/div[2]/div/div/div/div/div/ul/li[1]/a/div/p')
    temp.click()

    # Text
    temp = driver.find_element(By.ID, "post_type_link_text_only")
    temp.click()    

    logger.log('info', f'Preparing post..')
    # Tittle
    tittle = driver.find_element(By.XPATH, "//input[@placeholder='Add a title']")
    tittle.click()
    pyperclip.copy(Titletext)
    act = ActionChains(driver)
    act.key_down(Keys.CONTROL).send_keys("v").key_up(Keys.CONTROL).perform()

    # Text
    text = driver.find_element(By.XPATH, '//div[@contenteditable="true" and contains(@class, "ProseMirror remirror-editor")]')
    text.click()
    pyperclip.copy(Textdata)
    act = ActionChains(driver)
    act.key_down(Keys.CONTROL).send_keys("v").key_up(Keys.CONTROL).perform()

    logger.log('info', 'Loading file..')
    # File
    if Tier == 1:
        filename = f'Tier1_{datesave}.csv'
    elif Tier == 2:
        filename = f'Tier2_{datesave}.csv'
    elif Tier == 3:
        filename = f'Tier1_lstm_{datesave}.csv'

    upload_button = driver.find_element(By.XPATH, "//button[@data-tag='file-upload-button']")
    upload_button.click()
    time.sleep(2)
    pyautogui.hotkey("alt", "d")
    pyautogui.typewrite(ROOTPATH + '/' + PUBLISHPATH)
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.hotkey("alt", "n")
    pyautogui.typewrite(filename)
    pyautogui.hotkey("alt", "o")

    # Next
    temp = driver.find_element(By.XPATH, "//button//div[text()='Next']")
    temp.click()

    logger.log('info', f'Selecting viwers..')
    # Who can view
    temp = driver.find_element(By.ID, 'audience-selector')
    temp.click()

    # Select tiers
    temp = driver.find_element(By.XPATH, "//div[text()='Selected tiers']")
    temp.click()

    # Tier 1
    tier1box = driver.find_element(By.XPATH, "//input[@aria-label='Tier 1: Over/Under Predictions']")
    tier2box = driver.find_element(By.XPATH, "//input[@aria-label='Tier 2: Final Result Predictions']")
    tier3box = driver.find_element(By.XPATH, "//input[@aria-label='Tier 3: BetProphet.AI Precision Picks']")

    if Tier == 1:
        if tier1box.is_selected():
            pass
            logger.log('info', f'Tier1 selected for Tier{Tier}..')
        else:
            driver.execute_script("arguments[0].click();", tier1box)
            logger.log('info', f'Tier1 selected for Tier{Tier}..')

    elif Tier == 2:
        if tier1box.is_selected():
            driver.execute_script("arguments[0].click();", tier1box)
            logger.log('info', f'Tier1 de-selected for Tier{Tier}..')

        if tier2box.is_selected():
            pass
            logger.log('info', f'Tier2 selected for Tier{Tier}..')
        else:
            driver.execute_script("arguments[0].click();", tier2box)
            logger.log('info', f'Tier2 selected for Tier{Tier}..')
    
    elif Tier == 3:
         if tier1box.is_selected():
            driver.execute_script("arguments[0].click();", tier1box)
            logger.log('info', f'Tier1 de-selected for Tier{Tier}..')       
         if tier2box.is_selected():
            driver.execute_script("arguments[0].click();", tier2box)
            logger.log('info', f'Tier2 de-selected for Tier{Tier}..')      
        
         if tier3box.is_selected():
            pass
            logger.log('info', f'Tier3 selected for Tier{Tier}..')
         else:
            driver.execute_script("arguments[0].click();", tier3box)
            logger.log('info', f'Tier3 selected for Tier{Tier}..')


    # Publish
    temp = driver.find_element(By.XPATH, "//button//div[text()='Publish']")
    temp.click()  
   
    logger.log('info', f'Post published for Tier {Tier}..')
    sel.stop_server_and_driver(server, driver)
    return


def main():
    login_n_post(TIER1TITLE, TIER1TEXT, 1)
    login_n_post(TIER2TITLE, TIER2TEXT, 2)
    #login_n_post(TIER3TITLE, TIER3TEXT, 3)
    logger.log('info', f'Process completed..')
    return

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    ROOTPATH = os.path.dirname(__file__)
    _inita()
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
    main()
    try:
       pass
    except Exception as e:
        logger.log('critical', "Exception occured whie running", info=str(e))