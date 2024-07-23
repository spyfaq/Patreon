@echo
chdir /d "D:\Python Apps\Patreon"

echo Simulating Major leagues..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\AI_Predictions_major.py"

echo Simulating Minor leagues..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\AI_Predictions_minor.py"

echo Merging Simulations..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\predictions_merger.py"

echo Simulations are ready..