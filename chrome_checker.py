import os, platform, zipfile, requests, io, datetime, shutil
from jsonlogger_class import JSONLogger

LOGPATH = 'logs/chrome/'
LOGNAME = '{date}_chrome_logs'

def get_latest_chromedriver_version():
    """Fetch the latest ChromeDriver version."""
    url = "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE"
    response = requests.get(url)
    response.raise_for_status()
    version = response.text.strip()
    logger.log('info', f"Fetched latest ChromeDriver version: {version}")
    return version

def download_chromedriver(version):
    """Download the ChromeDriver for the current OS."""
    system = platform.system()
    if system == "Windows":
        system, driver_file = "win64", "chromedriver-win64.zip"
    elif system == "Linux":
        system, driver_file =  "linux64", "chromedriver-linux64.zip"
    elif system == "Darwin":  # macOS
        system, driver_file =  "mac-x64", "chromedriver-mac-x64.zip"
    else:
        raise ValueError(f"Unsupported OS: {system}")

    # Construct the download URL
    driver_url = f"https://storage.googleapis.com/chrome-for-testing-public/{version}/{system}/{driver_file}"
    logger.log('info', f"Downloading ChromeDriver {version} for {system}...")
    response = requests.get(driver_url)
    response.raise_for_status()

    # Extract the .exe file only into the working directory
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for file_name in z.namelist():
            if file_name.endswith('chromedriver.exe'):
                logger.log('info', f"Extracting {file_name} to working directory...")
                with z.open(file_name) as source_file:
                    with open('chromedriver.exe', "wb") as target_file:
                        target_file.write(source_file.read())
                logger.log('info', f"ChromeDriver saved as: {os.path.abspath('chromedriver.exe')}")
                break
        else:
            raise FileNotFoundError(f"chromedriver.exe not found in the archive.")



def main():
    try:
        latest_version = get_latest_chromedriver_version()
        logger.log('info', f"Latest ChromeDriver version: {latest_version}")
        download_chromedriver(latest_version)
    except Exception as e:
        logger.log('error', f"Error: {e}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    datesave = datetime.date.today().strftime('%Y%m%d')
    LOGNAME = LOGNAME.replace('{date}', datesave) + '.json'
    
    if os.path.exists(LOGPATH + '/' +LOGNAME):
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
    else:
        logger = JSONLogger(log_file=LOGNAME, log_dir=LOGPATH)
    main()