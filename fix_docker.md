# Fix Docker "dockerDesktopLinuxEngine" Error on Windows

## The Error
```
unable to get image: failed to connect to the docker API at 
npipe:////./pipe/dockerDesktopLinuxEngine
```

## Why This Happens
Docker Desktop is installed but the **Linux Engine (WSL2 backend) is not running**.

## Fix — 3 Steps

### Step 1: Open Docker Desktop
- Find the Docker whale icon 🐳 in your Windows taskbar (bottom-right system tray)
- If you don't see it, search "Docker Desktop" in the Start menu and open it
- **Wait** until the bottom-left status bar says "Engine running" (green dot)

### Step 2: Enable WSL2 Backend
If Docker opens but still fails:
1. Open Docker Desktop
2. Click the ⚙️ Settings gear icon
3. Go to **General**
4. Make sure **"Use the WSL 2 based engine"** is checked ✅
5. Click **Apply & Restart**

### Step 3: Try docker compose again
Once Docker Desktop shows "Engine running":
```powershell
docker compose up -d
```

## Alternative: Run Endee without Docker
If Docker keeps failing, you can use the Endee Cloud (free tier):
👉 https://endee.io/ — Sign up → Get cloud endpoint URL → Update .env:
```
ENDEE_BASE_URL=https://your-instance.endee.io
ENDEE_AUTH_TOKEN=your_token
```
