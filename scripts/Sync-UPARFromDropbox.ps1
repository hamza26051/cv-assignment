param(
    [Parameter(Mandatory=$true)]
    [string]$DropboxFolder,
    [string]$RepositoryFolder = "."
)
$ErrorActionPreference = "Stop"
$source = (Resolve-Path $DropboxFolder).Path
$repo = (Resolve-Path $RepositoryFolder).Path
if (-not (Test-Path (Join-Path $repo ".git"))) { throw "RepositoryFolder must be a cloned Git repository." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "Install Git before proceeding." }
if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) { throw "Install Git LFS before proceeding." }
git -C $repo lfs install
if ($LASTEXITCODE -ne 0) { throw "Git LFS initialization failed." }
# Copy every file from the complete Dropbox research collection; preserve all relative paths.
robocopy $source $repo /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /XD .git /XF Thumbs.db desktop.ini
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
# Binary model files must be transferred as LFS objects, not placeholder pointers.
git -C $repo lfs track "*.pt" "*.ts" "*.zip"
git -C $repo add -A
Write-Host "Reviewing staged changes:"
git -C $repo status --short
Write-Host "Reviewing Git LFS files:"
git -C $repo lfs ls-files
$answer = Read-Host "Commit and push these changes to origin/main? Type YES to proceed"
if ($answer -cne "YES") { Write-Host "No push performed."; exit 0 }
git -C $repo commit -m "Add complete UPAR research artifacts and model checkpoints"
if ($LASTEXITCODE -ne 0) { throw "Commit failed." }
git -C $repo push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed; verify GitHub permissions, LFS quota, and remote status." }
Write-Host "Push completed. Verify the repository and Git LFS file sizes on GitHub."
