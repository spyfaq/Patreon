#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time

DATAPATH = 'predictions_data/'
PUBLPATH = 'publish/'

listofpath = [DATAPATH, PUBLPATH]

def _housekeeping(days=15):
    print('Starting housekeeping..')
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
        print('Nothing to clean..')
    else:
        for data in result:
            if data['remove']:
                os.remove(data['file_path'])
                print('Housekeeping remove file..', data['file_path'])
    
    print('Housekeeping completed..')
    return()


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    try:
        _housekeeping()

    except Exception as e:
        print("CRITICAL: Exception occured whie running", e)