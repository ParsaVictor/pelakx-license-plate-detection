# ============================================================================
#  ساخت محیط ایزوله + کرنل ژوپیتر برای پروژه‌ی «تشخیص پلاک»
#  اجرا (یک بار):   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# ============================================================================

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$VenvDir    = Join-Path $ProjectDir ".venv"
$KernelName = "pelak"
$KernelLabel = "Python (pelak)"

Write-Host "پوشه‌ی پروژه: $ProjectDir" -ForegroundColor Cyan

# ۲) ساخت venv (اگر نبود)
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    $py = & py -3 -c "import sys; print(sys.executable)" 2>$null
    if (-not $py) { $py = "python" }
    Write-Host "پایتون پایه: $py" -ForegroundColor Green
    & $py -m venv $VenvDir
    Write-Host "venv ساخته شد: $VenvDir" -ForegroundColor Green
} else {
    Write-Host "venv از قبل هست — فقط به‌روزرسانی می‌شود." -ForegroundColor Yellow
}

$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

# ۳) نصب وابستگی‌ها  (تحمل اتصال ضعیف + torch از ایندکس رسمی CPU)
$pipOpts = @("--timeout", "120", "--retries", "10")
& $VenvPy -m pip install @pipOpts --upgrade pip wheel setuptools
& $VenvPy -m pip install @pipOpts torch==2.5.1 torchvision==0.20.1 `
    --index-url https://download.pytorch.org/whl/cpu
& $VenvPy -m pip install @pipOpts -r (Join-Path $ProjectDir "requirements.txt")

# ۴) ثبت کرنل ژوپیتر (برای Jupyter و VS Code یکسان است)
& $VenvPy -m ipykernel install --user --name $KernelName --display-name $KernelLabel

Write-Host ""
Write-Host "✅ تمام شد." -ForegroundColor Green
Write-Host "در ژوپیتر/VS Code کرنل «$KernelLabel» را انتخاب کن." -ForegroundColor Green
Write-Host "برای اجرای مستقیم: `"$VenvPy`" your_script.py" -ForegroundColor Green
