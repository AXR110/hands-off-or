@echo off
setlocal

pushd "%~dp0"
if errorlevel 1 exit /b 1

call npm start
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
