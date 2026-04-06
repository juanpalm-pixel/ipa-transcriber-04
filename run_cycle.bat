@echo off
setlocal

REM Always run from this .bat file's folder
cd /d "%~dp0"

echo ========================================
echo Running segmentation (pass 1)...
echo ========================================
python segmentation\run_segmentation.py || goto :fail
python segmentation\compare_models.py || goto :fail

echo ========================================
echo Running diarisation...
echo ========================================
python diarisation\run_diarisation.py || goto :fail
python diarisation\compare_models.py || goto :fail

echo ========================================
echo Running segmentation (pass 2)...
echo ========================================
python segmentation\run_segmentation.py || goto :fail
python segmentation\compare_models.py || goto :fail

echo.
echo Pipeline cycle completed successfully.
goto :end

:fail
echo.
echo ERROR: A step failed. Stopping pipeline.
exit /b 1

:end
endlocal
