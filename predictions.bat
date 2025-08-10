@echo
chdir /d "D:\Python Apps\Patreon"

echo Simulating Major leagues..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\majorleague_predictions.py"

echo Simulating Minor leagues..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\minorleague_predictions.py"

echo Merging Simulations..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\predictions_merger.py"

echo Creating tier's daily files..
"D:\Python Apps\Patreon\.venv\Scripts\python.exe" "D:\Python Apps\Patreon\Patreon_tier_.py"

echo Files are ready..