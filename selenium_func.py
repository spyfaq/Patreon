#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=Selenium helper functions=
Use this functions to start and stop the Selenium server in every other module.
"""

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

"""
Path where the Selenium driver for your browser is saved.
Go to http://selenium-python.readthedocs.io/installation.html to search for the
appropiate driver, copy it to the directory where this script is saved,
and change 'chromedriver' to match the name of your driver.
"""
PATH_TO_DRIVER = './chromedriver'

"""
Path where your web browser application is saved. This example is for MacOs, in Windows 7, 8,
and 10. Search Google if you don't know how to find your browser application's path.
"""

PATH_TO_BROWSER = r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'


"""
Functions
"""


def start_server_and_driver():
    """
    Start the Selenium server and driver and return them as objects.
    """

    # Initialize Chrome options
    chrome_options = Options()
    chrome_options.binary_location = PATH_TO_BROWSER
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-search-engine-choice-screen")

    # Set up capabilities
    capabilities = DesiredCapabilities.CHROME.copy()
    capabilities.update({
        'browserName': 'chrome',
        'version': '',
        'platform': 'ANY'
    })

    # Start the WebDriver server
    server = Service(PATH_TO_DRIVER)
    server.start()

    # Connect to the remote WebDriver
    driver = webdriver.Remote(
        command_executor=server.service_url,
        options=chrome_options,
        desired_capabilities=capabilities
    )

    driver.implicitly_wait(5)
    return server, driver


def stop_server_and_driver(server, driver):
    """
    Close the driver and then stop the server.
    =Args=
        driver: driver object returned by def start_server_and_driver()
        server: server object returned by def start_server_and_driver()
    """

    driver.close()
    server.stop()