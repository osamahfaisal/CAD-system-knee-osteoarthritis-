@REM @echo off
@REM rem Activate Python 3.11.7 environment
@REM start /min cmd /c "echo Activating Python 3.11.7 environment... && py -3.11 -m venv env && call env\Scripts\activate.bat && echo Running KOACAD.py... && python KOACAD.py && echo Deactivating Python environment... && deactivate && echo Project execution completed. Press any key to exit."
@REM pause >nul






@echo off
rem Activate Python 3.11.7 environment
echo Activating Python 3.11.7 environment...
py -3.11 -m venv setup\env
call setup\env\Scripts\activate.bat
rem Run the Python script
echo Running KOACAD.py...
python KOACAD.py
rem Deactivate Python environment
echo Deactivating Python environment...
deactivate
echo Project execution completed. Press any key to exit.
pause >nul
