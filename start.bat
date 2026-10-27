@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title WayPoint 旅行规划

set "POSTGRES_CONTAINER=travel_postgres"
set "POSTGRES_VOLUME=trip_postgres_data"
set "POSTGRES_DB=ai_travel_db"
set "POSTGRES_USER=travel_user"
set "POSTGRES_PASSWORD=travel123456"
set "REDIS_CONTAINER=travel_redis"
set "REDIS_VOLUME=trip_redis_data"
set "APP_URL=http://127.0.0.1:18000"
set "DOCS_URL=http://localhost:18000/ui/"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv，请先创建虚拟环境并安装依赖。
    pause
    exit /b 1
)

where docker >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 docker 命令，请先安装 Docker Desktop。
    pause
    exit /b 1
)

echo [1/6] 检查 Docker Desktop...
call :ensure_docker
if errorlevel 1 goto :failed

echo [2/6] 启动 PostgreSQL...
call :ensure_postgres
if errorlevel 1 goto :failed

echo [3/6] 启动 Redis...
call :ensure_redis
if errorlevel 1 goto :failed

echo [4/6] 初始化数据库表、Checkpointer、Store 和 pgvector...
.venv\Scripts\python.exe scripts\init_db.py
if errorlevel 1 (
    echo [错误] 数据库初始化失败，请查看上方日志。
    goto :failed
)

echo [5/6] 检查前端构建...
if exist "web\dist\index.html" (
    echo       前端已构建。
) else (
    where npm >nul 2>&1
    if errorlevel 1 (
        echo [警告] 未找到 npm，跳过前端构建；/ui 会显示构建提示。
    ) else (
        echo       首次运行，正在安装依赖并构建前端（约 1-3 分钟）...
        pushd web
        call npm install --no-audit --no-fund
        call npm run build
        popd
    )
)

echo [6/6] 启动后端服务...
echo       打开：%DOCS_URL%    API 文档：%APP_URL%/docs
echo       按 Ctrl+C 可停止后端；数据库容器会继续运行。

rem 在隐藏的 PowerShell 进程中等待健康检查通过，再打开浏览器。
start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$deadline=(Get-Date).AddSeconds(90); while((Get-Date) -lt $deadline){ try { Invoke-WebRequest -UseBasicParsing '%APP_URL%/' -TimeoutSec 2 | Out-Null; Start-Process '%DOCS_URL%'; exit 0 } catch { Start-Sleep -Seconds 1 } }"

.venv\Scripts\python.exe app\run.py
set "SERVER_EXIT=%errorlevel%"
if not "%SERVER_EXIT%"=="0" echo [错误] 后端服务异常退出，退出码：%SERVER_EXIT%
pause
exit /b %SERVER_EXIT%

:ensure_docker
docker info >nul 2>&1
if not errorlevel 1 (
    echo       Docker 已就绪。
    exit /b 0
)

echo       Docker 尚未运行，正在尝试启动 Docker Desktop...
set "DOCKER_DESKTOP_EXE=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
if not exist "%DOCKER_DESKTOP_EXE%" set "DOCKER_DESKTOP_EXE=%LOCALAPPDATA%\Docker\Docker Desktop.exe"
if not exist "%DOCKER_DESKTOP_EXE%" (
    echo [错误] 未找到 Docker Desktop，请手动启动后重新运行本脚本。
    exit /b 1
)

start "" "%DOCKER_DESKTOP_EXE%"
set /a DOCKER_TRIES=0
:wait_docker
docker info >nul 2>&1
if not errorlevel 1 (
    echo       Docker 已就绪。
    exit /b 0
)
set /a DOCKER_TRIES+=1
if !DOCKER_TRIES! geq 60 (
    echo [错误] Docker Desktop 在 120 秒内未就绪。
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto :wait_docker

:ensure_postgres
docker volume create "%POSTGRES_VOLUME%" >nul
if errorlevel 1 (
    echo [错误] 无法创建或读取 PostgreSQL 数据卷 %POSTGRES_VOLUME%。
    exit /b 1
)

docker container inspect "%POSTGRES_CONTAINER%" >nul 2>&1
if errorlevel 1 (
    echo       首次创建 PostgreSQL 容器...
    docker run -d --name "%POSTGRES_CONTAINER%" ^
        -e POSTGRES_DB="%POSTGRES_DB%" ^
        -e POSTGRES_USER="%POSTGRES_USER%" ^
        -e POSTGRES_PASSWORD="%POSTGRES_PASSWORD%" ^
        -p 15432:5432 ^
        -v "%POSTGRES_VOLUME%:/var/lib/postgresql/data" ^
        --restart unless-stopped ^
        pgvector/pgvector:pg17 >nul
) else (
    docker start "%POSTGRES_CONTAINER%" >nul
    if errorlevel 1 (
        echo       旧 PostgreSQL 容器端口映射不可用，保留数据卷并重建容器...
        docker rm "%POSTGRES_CONTAINER%" >nul
        if errorlevel 1 exit /b 1
        goto ensure_postgres
    )
)
if errorlevel 1 (
    echo [错误] PostgreSQL 容器启动失败，可能是容器配置或 5432 端口冲突。
    exit /b 1
)

set /a PG_TRIES=0
:wait_postgres
docker exec "%POSTGRES_CONTAINER%" pg_isready -U "%POSTGRES_USER%" -d "%POSTGRES_DB%" >nul 2>&1
if not errorlevel 1 (
    echo       PostgreSQL 已就绪，数据卷：%POSTGRES_VOLUME%。
    exit /b 0
)
set /a PG_TRIES+=1
if !PG_TRIES! geq 30 (
    echo [错误] PostgreSQL 在 60 秒内未就绪。
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto :wait_postgres

:ensure_redis
docker volume create "%REDIS_VOLUME%" >nul
if errorlevel 1 (
    echo [错误] 无法创建或读取 Redis 数据卷 %REDIS_VOLUME%。
    exit /b 1
)

docker container inspect "%REDIS_CONTAINER%" >nul 2>&1
if errorlevel 1 (
    echo       首次创建 Redis 容器...
    docker run -d --name "%REDIS_CONTAINER%" ^
        -p 6379:6379 ^
        -v "%REDIS_VOLUME%:/data" ^
        --restart unless-stopped ^
        redis:7-alpine redis-server --appendonly yes >nul
) else (
    docker start "%REDIS_CONTAINER%" >nul
)
if errorlevel 1 (
    echo [错误] Redis 容器启动失败，可能是容器配置或 6379 端口冲突。
    exit /b 1
)

set /a REDIS_TRIES=0
:wait_redis
set "REDIS_PING="
for /f "delims=" %%R in ('docker exec "%REDIS_CONTAINER%" redis-cli ping 2^>nul') do set "REDIS_PING=%%R"
if /i "!REDIS_PING!"=="PONG" (
    echo       Redis 已就绪，数据卷：%REDIS_VOLUME%。
    exit /b 0
)
set /a REDIS_TRIES+=1
if !REDIS_TRIES! geq 30 (
    echo [错误] Redis 在 60 秒内未就绪。
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto :wait_redis

:failed
echo.
echo 启动失败。请根据上方错误信息处理后重试。
pause
exit /b 1
