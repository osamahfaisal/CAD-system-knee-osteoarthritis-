@REM @echo off
@REM rem Activate Python 3.11.7 environment
@REM echo Activating Python 3.11.7 environment...
@REM start /min cmd /c "py -3.11 -m venv env && call env\Scripts\activate.bat && python -m pip install --upgrade pip && pip install -r requirements.txt && python KOACAD.py && deactivate && exit"
@REM echo Project execution completed.




@echo off
rem Activate Python 3.11.7 environment
echo Activating Python 3.11.7 environment...
py -3.11 -m venv setup\env
call setup\env\Scripts\activate.bat
rem Update pip
echo Updating pip...
python -m pip install --upgrade pip
rem Install requirements (only if not already installed)
if not exist setup/Right_Installed.txt (
    echo Installing requirements...
    pip install -r setup/requirements.txt
    if %errorlevel% equ 0 (
        echo Requirements installed successfully. > setup/Right_Installed.txt
    ) else (
        echo Failed to install requirements. Please try again.
        pause
        exit /b 1
    )
)

rem Run the Python script
echo Running KOACAD.py...
python KOACAD.py
rem Deactivate Python environment
echo Deactivating Python environment...
deactivate
echo Project execution completed. Press any key to exit.
pause >nul
