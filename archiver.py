#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, datetime, time
from jsonlogger_class import JSONLogger

DATAPATH = 'predictions_data/'
PUBLPATH = 'publish/'
LOGPATH = 'logs/'

listofpath = [DATAPATH, PUBLPATH, LOGPATH]

ARCHPATH = 'archiver/'
ARCHNAME = '{date}_archiver_logs'

def _housekeeping(days=15):
    logger.log('info', 'Starting housekeeping..')
    current_time = time.time()
    threshold_days = days * 24 * 3600  # days in seconds

    result = []
    tobedeleted = 0
    for datapath in listofpath:
        for root, dirs, files in os.walk(datapath):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                last_modified = os.path.getmtime(file_path)
                is_old = (current_time - last_modified) > threshold_days

                entry = {
                    "file_path": file_path,
                    "last_modified_date": last_modified,
                    "remove": is_old
                }

                if is_old:
                    tobedeleted += 1
                result.append(entry)

    if tobedeleted == 0:
        logger.log('info', 'Nothing to clean..')
    else:
        for data in result:
            if data['remove']:
                os.remove(data['file_path'])
                logger.log('info', 'Housekeeping remove file..', info=str(data['file_path']))
    
    logger.log('info', 'Housekeeping completed..')
    return()


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    datesave = datetime.date.today().strftime('%Y%m%d')
    ARCHNAME = ARCHNAME.replace('{date}', datesave) + '.json'
    
    if os.path.exists(ARCHPATH + '/' +ARCHNAME):
        logger = JSONLogger(log_file=ARCHNAME, log_dir=ARCHPATH)
    else:
        logger = JSONLogger(log_file=ARCHNAME, log_dir=ARCHPATH)

    try:
        _housekeeping()

    except Exception as e:
        logger.log('critical', "Exception occured whie running", info=str(e))