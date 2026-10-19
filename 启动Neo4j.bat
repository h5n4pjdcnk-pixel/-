@echo off
title Neo4j Console
if "%NEO4J_HOME%"=="" (
    echo [ERROR] NEO4J_HOME is not set.
    echo Set it to your local Neo4j installation directory, for example:
    echo   set NEO4J_HOME=%USERPROFILE%\neo4j
    pause
    exit /b 1
)
if not "%JAVA_HOME%"=="" echo JAVA_HOME=%JAVA_HOME%
echo Starting Neo4j from "%NEO4J_HOME%" ...
cd /d "%NEO4J_HOME%"
call bin\neo4j.bat console
echo.
echo Neo4j has stopped.
pause