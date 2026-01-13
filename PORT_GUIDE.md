# Port Allocation Guide for Your 3 Repos

## Current Status

**SelfEngine** (this repo): **CLI Tool** - Does NOT use a web server/port

Your other **2 repos** can use these available localhost ports:

---

## Currently In-Use Localhost Ports ⚠️

```
4381, 5354, 6463, 9080, 9180, 12700, 14249, 14630,
26822, 30865, 32683, 33683, 49221, 52138, 53314,
54002, 55900, 57506, 60979, 61988, 62196, 64138
```

---

## Available Ports for Your Repos ✅

### Common Development Ports (Recommended)

| Port | Status | Recommended For |
|------|--------|-----------------|
| **3000** | 🟢 FREE | Next.js, React (primary) |
| **3001** | 🟢 FREE | Next.js, React (secondary) |
| **3002** | 🟢 FREE | Next.js, React (tertiary) |
| **5000** | 🟢 FREE | Flask, Python backend |
| **5001** | 🟢 FREE | Flask, Python backend (secondary) |
| **8000** | 🟢 FREE | Django, Python backend |
| **8001** | 🟢 FREE | Django, Python backend (secondary) |
| **8080** | 🟢 FREE | General purpose |
| **8081** | 🟢 FREE | General purpose (secondary) |

---

## Suggested Setup for Your 3 Repos

```
┌─────────────────────────────────────┐
│  SelfEngine (This Repo)             │
│  • CLI Tool                         │
│  • No port needed                   │
│  • Run: python -m cli.main "..."   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Repo #2 (Prompty/Nexus?)          │
│  • Port: 3000 (primary)             │
│  • Run: npm run dev                 │
│  • URL: http://localhost:3000       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Repo #3 (vibe-guard-official?)    │
│  • Port: 3001 (secondary)           │
│  • Run: npm run dev -- --port 3001  │
│  • URL: http://localhost:3001       │
└─────────────────────────────────────┘
```

---

## Quick Reference: Port Usage Command

### Node.js / Next.js
```bash
# Default (3000)
npm run dev

# Custom port
npm run dev -- --port 3001
# OR
PORT=3001 npm run dev
```

### Python / Flask
```bash
# Default (5000)
python app.py

# Custom port
python app.py --port 5001
# OR
FLASK_ENV=development FLASK_APP=app.py flask run --port 5001
```

### Python / Django
```bash
# Default (8000)
python manage.py runserver

# Custom port
python manage.py runserver 8001
```

---

## If You Need to Kill Existing Port

```bash
# Find process using a port (e.g., 3000)
netstat -ano | findstr ":3000"

# Kill by PID (e.g., PID 1234)
taskkill /PID 1234 /F

# Or in PowerShell (safer)
Get-Process | Where-Object {$_.Handles -like "*3000*"} | Stop-Process -Force
```

---

## Safe Port Ranges

| Range | Use Case |
|-------|----------|
| **1024-49151** | User ports (safe) |
| **49152-65535** | Dynamic/ephemeral ports (very safe) |
| **8000-9000** | Development servers |
| **3000-3999** | Frontend development |
| **5000-5999** | Backend services |

---

## Summary

✅ **SelfEngine**: No port (CLI tool)
🟢 **Repo #2**: Use port **3000**
🟢 **Repo #3**: Use port **3001**

All ports are currently free and ready to use!
