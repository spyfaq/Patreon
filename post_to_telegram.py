#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_to_telegram.py

Posts the day's published files to Telegram, reading them from THIS
REPOSITORY rather than Dropbox.

Replaces post_from_dropbox.py. The pipeline commits everything it
produces into history/<date>/, so the files are already here on checkout
-- listing and downloading them from Dropbox was a round-trip to fetch
files this repo already had, and it meant three secrets
(DROPBOX_APP_KEY/APP_SECRET/REFRESH_TOKEN) plus a token-exchange step in
two separate workflows just to read our own output.

Looks in history/<date>/ first (where the pipeline commits) and falls
back to publish/ (where a run that just generated them still has them on
disk), so this works both as a separate scheduled job and inline right
after generation.
"""

import os
import re
import requests

import date_utils

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_IDS = {
    "Public": os.environ["TELEGRAM_CHAT_ID_TIER1"],
    "VIP": os.environ["TELEGRAM_CHAT_ID_TIER3"],
}

HISTORYPATH = 'history'
PUBLISHPATH = 'publish'

TELEGRAM_MAX_LEN = 4096

# Fetch and label windows are aligned (see date_utils.py): a run on day X
# should only ever produce files dated day X. Tomorrow's date is still
# checked as a defensive safety net -- cheap, and silently posting
# nothing is worse than one lookup that finds nothing.
CANDIDATE_DATE_STRS = date_utils.relevant_date_strs()


def candidate_paths(filename, date_str):
    """Where a published file could be, most-authoritative first."""
    return [
        os.path.join(HISTORYPATH, date_str, filename),
        os.path.join(PUBLISHPATH, filename),
    ]


def find_file(filename, date_str):
    for path in candidate_paths(filename, date_str):
        if os.path.exists(path):
            return path
    return None


def markdown_to_telegram_html(text):
    """Convert the Best Bets markdown into Telegram-flavoured HTML.

    The file itself stays markdown because it is committed to the repo,
    where GitHub renders it properly. Telegram has no markdown headings
    at all and its legacy Markdown mode uses single asterisks, so posting
    the raw file with parse_mode=HTML showed literal '##' and '**' to
    subscribers. Converting on send keeps both readable.
    """
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in ('---', '***'):
            out.append('')
            continue
        # Headings -> bold. Deepest first so '##' isn't caught by the '#' rule.
        m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if m:
            out.append(f"<b>{escape_html(m.group(2))}</b>")
            continue
        out.append(inline_markdown(escape_html(line)))
    text = "\n".join(out)
    # Collapse the runs of blank lines the heading/rule handling leaves behind.
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def escape_html(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def inline_markdown(s):
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'<i>\1</i>', s)
    s = re.sub(r'^\s*-\s+', '• ', s)
    return s


def send_telegram_message(chat_id, text):
    """Send a message, splitting into <=4096-char chunks (Telegram rejects
    longer text outright) and checking the response rather than assuming
    success."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    chunks = [text[i:i + TELEGRAM_MAX_LEN] for i in range(0, len(text), TELEGRAM_MAX_LEN)] or [text]

    for chunk in chunks:
        resp = requests.post(url, data={
            "chat_id": chat_id, "text": chunk,
            "parse_mode": "HTML", "disable_web_page_preview": True,
        })
        result = {}
        try:
            result = resp.json()
        except ValueError:
            pass
        if not resp.ok or not result.get("ok", False):
            print(f"Telegram sendMessage failed (status {resp.status_code}): "
                  f"{result.get('description', resp.text)}")
            return False
    return True


def send_telegram_document(chat_id, path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    with open(path, 'rb') as f:
        resp = requests.post(url, data={"chat_id": chat_id},
                             files={"document": (os.path.basename(path), f.read())})
    result = {}
    try:
        result = resp.json()
    except ValueError:
        pass
    if not resp.ok or not result.get("ok", False):
        print(f"Telegram sendDocument failed (status {resp.status_code}): "
              f"{result.get('description', resp.text)}")
        return False
    return True


def post_text(tier, filename, date_str, convert_markdown=False):
    path = find_file(filename, date_str)
    if not path:
        return False
    content = open(path, encoding='utf-8').read()
    if convert_markdown:
        content = markdown_to_telegram_html(content)
    if send_telegram_message(CHAT_IDS[tier], content):
        print(f'Posted {filename} to {tier}.')
        return True
    print(f'Failed to post {filename} to {tier}.')
    return False


def post_document(tier, filename, date_str):
    path = find_file(filename, date_str)
    if not path:
        return False
    if send_telegram_document(CHAT_IDS[tier], path):
        print(f'Posted {filename} to {tier}.')
        return True
    print(f'Failed to post {filename} to {tier}.')
    return False


def main():
    posted_any = False

    for date_str in sorted(CANDIDATE_DATE_STRS):
        # Each of these is posted at most once per date. A date with no
        # files at all is normal -- CANDIDATE_DATE_STRS deliberately
        # includes tomorrow as a safety net.
        results = [
            post_text('Public', f'Public_{date_str}.txt', date_str),
            post_text('VIP', f'VIP_{date_str}.txt', date_str),
            post_document('VIP', f'VIP_{date_str}.xlsx', date_str),
            # Best Bets is markdown -- convert before sending.
            post_text('VIP', f'BestBets_{date_str}.txt', date_str, convert_markdown=True),
        ]
        if any(results):
            posted_any = True

    if not posted_any:
        print(f'Nothing to post. Looked for dates {sorted(CANDIDATE_DATE_STRS)} '
              f'in {HISTORYPATH}/<date>/ and {PUBLISHPATH}/.')


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) or '.')
    main()
