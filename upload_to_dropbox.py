#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import dropbox

# Get Dropbox access token from environment variable
DROPBOX_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")
if not DROPBOX_TOKEN:
    raise ValueError("DROPBOX_ACCESS_TOKEN not found in environment variables.")

# Local folder with files to upload
LOCAL_FOLDER = "publish/"
DROPBOX_FOLDER = "/telegram_content"

# Initialize Dropbox client
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

# Loop through files in the local folder
for root, _, files in os.walk(LOCAL_FOLDER):
    for filename in files:
        local_path = os.path.join(root, filename)
        dropbox_path = f"{DROPBOX_FOLDER}/{filename}"

        with open(local_path, "rb") as f:
            try:
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
                print(f"Uploaded {filename} to {dropbox_path}")
            except Exception as e:
                print(f"Failed to upload {filename}: {e}")
