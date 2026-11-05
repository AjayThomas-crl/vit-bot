@echo off
REM One-click pipeline: download posts + comments, clean, build topic report.
REM Progress goes to logs\run.log. Safe to close and re-run (it resumes).
cd /d "%~dp0"
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
set LOG=logs\run.log
echo ===== START %date% %time% ===== >> %LOG%
call .venv\Scripts\activate.bat
python -u scrape_vit.py --only posts >> %LOG% 2>&1
python -u clean.py >> %LOG% 2>&1
python -u explore.py >> %LOG% 2>&1
echo ===== POSTS STAGE DONE %date% %time% ===== >> %LOG%
python -u scrape_vit.py --only comments >> %LOG% 2>&1
python -u clean.py >> %LOG% 2>&1
python -u explore.py >> %LOG% 2>&1
echo ===== ALL DONE %date% %time% ===== >> %LOG%
