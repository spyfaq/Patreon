@echo
chdir /d "D:\Python Apps\Patreon"
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\tips_prep.py"

echo Data are ready to be published..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\tips_posting.py"

echo Data published..