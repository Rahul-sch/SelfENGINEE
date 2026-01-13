# Multi-Repo Development Setup Guide

## Your 3 Repos

### 1. SelfEngine (CLI Tool - NO PORT)
**Location:** `c:\Users\rahul\Desktop\SelfEngine`

```bash
# Just run commands directly - no web server
python -m cli.main "Write a fibonacci function"
python -m cli.main --use-sabs "Write a file reader"
python -m pytest tests/ -v
```

**Type:** Command-line application
**Port:** ❌ None needed

---

### 2. Prompty/Nexus (Next.js Frontend)
**Location:** `c:\Users\rahul\Desktop\Prompty\nexus\apps\web`

```bash
# Navigate to project
cd "c:\Users\rahul\Desktop\Prompty\nexus\apps\web"

# Install dependencies
npm install
# OR
pnpm install

# Run on port 3000 (default)
npm run dev

# URL: http://localhost:3000
```

**Type:** Next.js application
**Port:** 🟢 **3000** (primary)
**Package Manager:** npm or pnpm

---

### 3. Vibe Guard (Node.js Backend/App)
**Location:** `c:\Users\rahul\Desktop\vibe-guard-official`

```bash
# Navigate to project
cd "c:\Users\rahul\Desktop\vibe-guard-official"

# Install dependencies
npm install
# OR
pnpm install

# Run on custom port
PORT=3001 npm run dev
# OR
npm run dev -- --port 3001

# URL: http://localhost:3001
```

**Type:** Node.js/Next.js application
**Port:** 🟢 **3001** (secondary)
**Package Manager:** npm or pnpm

---

## Quick Start - All 3 Repos

### Terminal 1: SelfEngine (CLI)
```bash
cd "c:\Users\rahul\Desktop\SelfEngine"
python -m pytest tests/ -v
# OR run a generation task
python -m cli.main "Write a simple calculator"
```

### Terminal 2: Prompty (Port 3000)
```bash
cd "c:\Users\rahul\Desktop\Prompty\nexus\apps\web"
npm install && npm run dev
# Runs on http://localhost:3000
```

### Terminal 3: Vibe Guard (Port 3001)
```bash
cd "c:\Users\rahul\Desktop\vibe-guard-official"
npm install && PORT=3001 npm run dev
# Runs on http://localhost:3001
```

---

## Port Summary

| Repo | Port | URL | Command |
|------|------|-----|---------|
| SelfEngine | None | CLI | `python -m cli.main` |
| Prompty | 3000 | http://localhost:3000 | `npm run dev` |
| Vibe Guard | 3001 | http://localhost:3001 | `PORT=3001 npm run dev` |

---

## If Ports Conflict

### Check what's using a port
```bash
netstat -ano | findstr ":3000"
netstat -ano | findstr ":3001"
```

### Kill a process by port (PowerShell)
```powershell
$port = 3000
$process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $process.OwningProcess -Force
    Write-Host "Killed process on port $port"
}
```

### Kill a process by port (CMD)
```bash
taskkill /F /PID [PID_NUMBER]
```

---

## Alternative Ports (If Needed)

All these are FREE and available:
- **3002, 3003, 3004** - Frontend alternatives
- **5000, 5001** - Backend alternatives
- **8000, 8001, 8080, 8081** - General purpose

---

## Development Workflow

```
┌──────────────────────────────────────┐
│  SelfEngine                          │
│  Research-grade Code Generation      │
│  python -m cli.main "..."            │
└──────────────────────────────────────┘
           ↑
           │ (if you integrate)
           ↓
┌──────────────────────────────────────┐
│  Prompty/Nexus (3000)                │
│  Frontend UI                         │
│  npm run dev                         │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│  Vibe Guard (3001)                   │
│  Backend/Service                     │
│  PORT=3001 npm run dev               │
└──────────────────────────────────────┘
```

---

## Tips

✅ Use separate terminal windows for each repo
✅ Keep SelfEngine as a CLI tool (no port needed)
✅ Frontend on 3000, Backend on 3001 = clean separation
✅ All ports are currently free
✅ Can easily switch ports if needed

Good to go! 🚀
