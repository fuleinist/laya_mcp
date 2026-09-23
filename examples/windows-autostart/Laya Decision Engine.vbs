' Laya System-1 decision engine (hidden, idempotent) - starts on Windows login.
' Drop this in: %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
' Run style 0 = no console window, and children inherit the hidden console.
' Edit the path if you put laya-serve.cmd somewhere else.
CreateObject("WScript.Shell").Run """G:\dev\AI\laya\laya-serve.cmd""", 0, False