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
LOGNAME = '{date}_accuracypost_logs'
PUBLISHPATH = 'publish/'

def _inita():
    global USERMAIL, USERPASS, AccTITLE, AccTEXT
    with open('init.cfg', 'r', encoding='utf-8') as FIL:
        data = FIL.readlines()
        FIL.close()
    
    USERMAIL = data[0].split(' = ')[-1].strip()
    USERPASS = data[1].split(' = ')[-1].strip()

    random_accuracy_post = random.choice(bettexts.accuracy_posts)
    AccTITLE = random_accuracy_post['Title']
    AccTEXT = random_accuracy_post['Text']

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

    # Type Text
    temp = driver.find_element(By.ID, "post_type_link_image_file")
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

    # Files
    tier_acc = f'"tiers_accuracy_plot_{datesave}.png" '
    preds_acc = f'"predictions_accuracy_plot_{datesave}.png"'
    divs_acc = f'"division_accuracy_plot_{datesave}.png"'

    upload_button = driver.find_element(By.XPATH, "//button[@data-tag='file-upload-button']")
    upload_button.click()
    time.sleep(2)
    pyautogui.hotkey("alt", "d")
    pyautogui.typewrite(ROOTPATH + '/' + PUBLISHPATH)
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.hotkey("alt", "n")
    pyautogui.typewrite(tier_acc)
    pyautogui.typewrite(preds_acc)
    pyautogui.typewrite(divs_acc)
    pyautogui.hotkey("alt", "o")

    time.sleep(10)
    # Next
    temp = driver.find_element(By.XPATH, "//button//div[text()='Next']")
    temp.click()

    logger.log('info', f'Selecting viwers..')
    # Who can view
    temp = driver.find_element(By.XPATH, "//button[@aria-label='Who can view this post']")
    temp.click()

    # make it public
    temp = driver.find_element(By.XPATH, "//div[text()='Public']")
    temp.click()

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
    login_n_post(AccTITLE, AccTEXT, 0)
    logger.log('info', f'Process completed..')
    return

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    ROOTPATH = os.path.dirname(__file__)
    _inita()
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)

    try:
        #main()
        logger.log('warning', "We do not post accuracy any more")
    except Exception as e:
        logger.log('critical', "Exception occured whie running", info=str(e))