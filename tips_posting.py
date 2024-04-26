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
PUBLISHPATH = 'D:/Python Apps/Patreon/publish/'

def _inita():
    global USERMAIL, USERPASS, TIER1TITLE, TIER1TEXT, TIER2TITLE, TIER2TEXT
    with open('init.cfg', 'r', encoding='utf-8') as FIL:
        data = FIL.readlines()
        FIL.close()
    
    USERMAIL = data[0].split(' = ')[-1].strip()
    USERPASS = data[1].split(' = ')[-1].strip()

    random_tier_1_post = random.choice(bettexts.tier_1_posts)
    random_tier_2_post = random.choice(bettexts.tier_2_posts)

    TIER1TITLE = random_tier_1_post['Title']
    TIER1TEXT = random_tier_1_post['Text']
    TIER2TITLE = random_tier_2_post['Title']
    TIER2TEXT = random_tier_2_post['Text']

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

    # open selection menu
    temp = driver.find_element(By.XPATH, "//button[@aria-label='Account menu']")
    temp.click()

    # move to creator page
    temp = driver.find_element(By.XPATH, "//a[@href='/user']")
    temp.click()

    # close pop up
    try:
        temp = driver.find_element(By.XPATH, "//button[@aria-label='Close Dialog']")
        temp.click()
    except:
        pass
    logger.log('info', 'Creator page loaded..')

    try:
        temp = driver.find_element(By.XPATH, "//button[@aria-label='Close']")
        temp.click()
    except:
        pass

    # Create
    temp = driver.find_element(By.XPATH, "/html/body/div/div/div[2]/div/div/nav/div[2]/div/div[2]/div[2]/div/button")
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
    text = driver.find_element(By.XPATH, '/html/body/div/div/div[4]/div/main/div[1]/div/div/div/div/div/div[1]/div/div[3]/div')
    text.click()
    pyperclip.copy(Textdata)
    act = ActionChains(driver)
    act.key_down(Keys.CONTROL).send_keys("v").key_up(Keys.CONTROL).perform()

    logger.log('info', 'Loading file..')
    # File
    if Tier == 1:
        filename = f'Tier1_{datesave}.csv'
    else:
        filename = f'Tier2_{datesave}.csv'


    upload_button = driver.find_element(By.XPATH, "//button[@data-tag='file-upload-button']")
    upload_button.click()
    time.sleep(2)
    pyautogui.hotkey("alt", "d")
    pyautogui.typewrite(PUBLISHPATH)
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
    temp = driver.find_element(By.XPATH, '/html/body/div/div/div[4]/div/main/div[1]/div/div/div/div/div/div/div[1]/div[2]/div/div/div[2]/button/button/div/div[1]/div/div')
    temp.click()

    # Select tiers
    temp = driver.find_element(By.XPATH, "//div[text()='Select tiers']")
    temp.click()

    # Tier 1
    tier1box = driver.find_element(By.XPATH, "//input[@aria-label='Tier 1: Over/Under Predictions']")
    tier2box = driver.find_element(By.XPATH, "//input[@aria-label='Tier 2: All-Inclusive Access']")

    if Tier == 1:
        if tier1box.is_selected():
            pass
            logger.log('info', f'Tier1 selected for Tier{Tier}..')
        else:
            driver.execute_script("arguments[0].click();", tier1box)
            logger.log('info', f'Tier1 selected for Tier{Tier}..')

    else:
        if tier1box.is_selected():
            driver.execute_script("arguments[0].click();", tier1box)
            logger.log('info', f'Tier1 de-selected for Tier{Tier}..')

        if tier2box.is_selected():
            pass
            logger.log('info', f'Tier2 selected for Tier{Tier}..')
        else:
            driver.execute_script("arguments[0].click();", tier2box)
            logger.log('info', f'Tier2 selected for Tier{Tier}..')


    # Publish
    temp = driver.find_element(By.XPATH, "//button//div[text()='Publish']")
    temp.click()  

    # Close share pop up
    temp = driver.find_element(By.XPATH, "//button[@aria-label='Close the share dialog']")
    temp.click() 
    
    logger.log('info', f'Post published for Tier {Tier}..')
    sel.stop_server_and_driver(server, driver)
    return


def main():
    login_n_post(TIER1TITLE, TIER1TEXT, 1)
    login_n_post(TIER2TITLE, TIER2TEXT, 2)
    logger.log('info', f'Process completed..')
    return

if __name__ == '__main__':
    _inita()
    os.chdir('D:\\Python Apps\\Patreon')
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        main()
    except Exception as e:
        logger.log('critical', "Exception occured whie running", info=str(e))