# Запуск бота Coach Grebenyuk
Set-Location $PSScriptRoot

# Создаём venv если нет
if (-not (Test-Path ".venv")) {
    Write-Host "Создаю виртуальное окружение..."
    python -m venv .venv
}

# Активируем и устанавливаем зависимости
.\.venv\Scripts\pip install -r requirements.txt -q

# Загружаем .env
Get-Content .env | ForEach-Object {
    if ($_ -match "^([^#][^=]+)=(.+)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim())
    }
}

Write-Host "Запускаю бота..."
.\.venv\Scripts\python bot.py
