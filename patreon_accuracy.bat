@echo
chdir /d D:\Python Apps\Patreon
D:\Python37\python.exe "D:\Python Apps\Patreon\accuracy_prep.py"

echo Data are ready to be published..
D:\Python37\python.exe "D:\Python Apps\Patreon\accuracy_posting.py"

echo Data published..