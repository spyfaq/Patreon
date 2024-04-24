@echo
chdir /d D:\Python Apps\Patreon
D:\Python37\python.exe "D:\Python Apps\Patreon\tips_prep.py"

echo Data are ready to be published..
D:\Python37\python.exe "D:\Python Apps\Patreon\tips_posting.py"

echo Data published..