@echo
chdir /d "D:\Python Apps\Patreon"

echo Creating Tiers..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\Patreon_tier_.py"

echo Posting to Patreon..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\API_Patreon_post.py"

echo Posts are online..