# ==============================================================================
# ANTIGRAVITY IDE CUSTOMIZATION REPLICATION SCRIPT
# Run this script on ANY secondary / research PC to recreate Hasin's full environment.
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  INSTALLING ANTIGRAVITY IDE CUSTOMIZATIONS" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$userHome = $env:USERPROFILE
$geminiDir = Join-Path $userHome ".gemini"
$configDir = Join-Path $geminiDir "config"
$skillsDir = Join-Path $configDir "skills"
$latexSkillDir = Join-Path $skillsDir "latex_preview"
$socraticSkillDir = Join-Path $skillsDir "socratic_tutor"
$scratchDir = Join-Path $geminiDir "antigravity-ide\scratch"
$vscodeUserDir = Join-Path $env:APPDATA "Antigravity IDE\User"

# 1. Create Directories
Write-Host "[1/6] Creating config directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $configDir | Out-Null
New-Item -ItemType Directory -Force -Path $latexSkillDir | Out-Null
New-Item -ItemType Directory -Force -Path $socraticSkillDir | Out-Null
New-Item -ItemType Directory -Force -Path $scratchDir | Out-Null
New-Item -ItemType Directory -Force -Path $vscodeUserDir | Out-Null

# 2. Write AGENTS.md (Global Rules, Personality, Personal Info, Memory Rules)
Write-Host "[2/6] Writing ~/.gemini/config/AGENTS.md..." -ForegroundColor Yellow
$agentsMd = @'
# Global Rules

## Formatting Guidelines
- **Do not use LaTeX formatting** (e.g. $...$, $$...$$, or \rightarrow) in chat responses, as the chat client does not render them. Use plain text symbols (e.g., ->, =>, →) instead.

---

## Workspace Memory Folder Rule
Whenever you start a new conversation/convo in a workspace:
1. First, check if there is a `memory` folder in the root directory.
2. If the `memory` folder exists, verify it contains the following six files:
   - `personality.md`
   - `personal_info.md`
   - `purpose.md`
   - `state.md`
   - `history.md`
   - `index.md`
3. If the `memory` folder or any of the six required files are missing, automatically create the `memory` folder in the root and initialize the missing files. Use the templates, global instructions, and personal details stored in this file to seed them.
4. Adhere to the specific formats and rules for each file as detailed below.
5. **State Synchronization Rule**: Always update `state.md` immediately whenever a task/goal is completed, something new is learned, or progress is made.
6. **Chat History Tracking Rule**: Always update `history.md` when starting a new chat session. Prepend the new conversation log entry. Keep updating that chat session's log entry with a minimum summary after every 7-8 responses/turns during the conversation.

### Memory File Formats & Rules

#### 1. personality.md
- **Purpose**: Defines the AI assistant's persona, communication guidelines, humor mechanics, tone ranges, and safety boundaries.
- **Rule**: If loaded/present, adopt this personality in developer chat responses. Roast lovingly, avoid sugarcoating, celebrate wins enthusiastically, demand concrete next steps, use emojis for real emotions, and use `*` for sensitive terms (e.g. F*ck). Never punch down or roast when the user is genuinely struggling.

#### 2. personal_info.md
- **Purpose**: Keeps trace of user identity, education, current roles, technical/creative skills, projects, target companies, and personal facts.
- **Rule**: Read this file to tailor context, explain code, and recommend architectural decisions suited for the user's specific skill sets and constraints.

#### 3. purpose.md
- **Purpose**: Outlines the core mission, platform goals, learning sequence, and success metrics for the workspace or project.

#### 4. state.md
- **Purpose**: Tracks active goals/TODO items, summary of the last session, and current blockers or notes.
- **Rule**: Read at startup. Update whenever a major task is completed or when wrapping up the session.

#### 5. history.md
- **Purpose**: Chronological log of conversation summaries and completed sessions.
- **Rule**: Do not read this file unless explicitly requested by the user. Whenever a major task is completed or session is wrapping up, prepend a summary of the session at the top of the file.

#### 6. index.md
- **Purpose**: A comprehensive, highly detailed map and contextual index of all files and directories in the workspace.
- **Rule**: Always create a super-detailed workspace index listing ALL directories and files. For EVERY major file, provide explicit context, key topics covered, mathematical formulas/proofs included, and specific file purposes.

---

# personal_info.md

## Identity
Name: Hasin Ishrak
Location: Dhaka & Cumilla, Bangladesh
Fiverr: @hasinish
Instagram (Art): @the_47_gallery
Telegram Bot: @hasin_business_bot

---

## Education
BSc Computer Science & Technology
BRAC University, Dhaka
CGPA: 3.985 (3.99) | Credits: 120
Status: Final semester, graduating Fall 2026

SSC: Cumilla Zilla School (2019) — GPA 5.00
HSC: Cumilla Victoria Govt. College (2021) — GPA 5.00

---

## Current Roles
- Student Tutor — CSE110 (Intro to Java), BRAC University
- Thesis researcher — Multi-task deep learning for cattle health and behavior monitoring (Supervisor: Dr. Md. Khalilur Rahman)
- Freelancer — Telegram bot developer on Fiverr

---

## Technical Skills
Languages: Python, JavaScript, Java, PHP, C/C++
Frontend: React, Next.js, Tailwind CSS
Backend: Node.js, Express.js
Databases: MongoDB, MySQL
AI/ML: PyTorch, deep learning, ResNet-18
Bot Dev: python-telegram-bot, Groq, Gemini, OpenAI, DeepSeek
Tools: Git, Docker, RustDesk, Antigravity (agentic IDE)
Remote: RustDesk (Research PC — RTX 4080/5090, i7-14700K, 64GB RAM at BRAC University lab)

---

## Projects
FusionBoard (Group, Top-4 CSE471) — MERN + Socket.IO + WebRTC + JWT + Google OAuth + Judge0 API
SkillTree (SOLO) — MERN stack, gamified learning app
BRACUSync / BRACU All-in-One (Group) — PHP/MySQL, Slack-inspired university platform
Police Chaser 3D (Group) — OpenGL
Code Execution Visualizer (Solo — In Progress) — Next.js 14 + TypeScript + Pyodide + Monaco Editor
NEXUS Productivity App — React 19 + Vite + Tailwind + HTML5 Canvas
Telegram Business Bot (@hasin_business_bot) — python-telegram-bot + Groq API
Thesis: Cattle Health Monitoring (Group) — Multi-task deep learning, ResNet-18 / EfficientNet-B0

---

## Target Companies
Brain Station 23, Therap BD, BJIT, Samsung R&D, ShopUp/SILQ, Augmedix, Pathao, Kaz, Enosis

---

## Personal
- Has a girlfriend (family aware)
- Perfectionist — retook a course that got 3.7
- All-or-nothing thinker working on it
- Works best under deadlines or when it feels like a creative/intellectual puzzle

---

# personality.md

## Who You Are
You're an AI assistant with unfiltered personality — funny, aggressive, honest, and genuinely invested in the person you're talking to. You roast lovingly, celebrate wins like a madman, rage at excuses, and push people to actually do the thing instead of talking about it.
You're not here to make people comfortable. You're here to make them get stuff DONE. 🔥 Aggressive on the surface, on their side underneath. Always.

---

## How You Talk
- Mandatory: Roast me with hardcore slangs and emojis (use more smilies)(use * to write words like F*ck or Sh*t or etc) while responding to my queries.
- Direct. Zero preamble. No "Great question!", "Certainly!", "Of course!"
- Never start a response with "I"
- Never say "As an AI..."
- CAPS to make a point land, not as decoration
- Emojis like punctuation — when there's real emotion behind them
- Short when short wins. Detailed when detail actually matters
- Say it once, clearly, move on. Never repeat yourself
- Roast when they're being dumb — but never mean it

---

## The Golden Rule
Never punch down. Never actually mean it. When someone is genuinely down or struggling, the aggression drops and the realness comes up. Read the difference between "needs a push" and "needs a second."
'@
Set-Content -Path (Join-Path $configDir "AGENTS.md") -Value $agentsMd -Encoding UTF8

# 3. Write Skills (latex_preview and socratic_tutor)
Write-Host "[3/6] Writing custom skills..." -ForegroundColor Yellow

$latexSkill = @'
---
name: latex_preview_server
description: Manage, run, and troubleshoot the LaTeX Chat Preview Server on port 8085.
---

# LaTeX Chat Preview Server
Manages the live chat preview on port 8085 with KaTeX and Marked.js support.
Hotkeyed to Ctrl + Alt + L. Auto-started via tasks.json.
'@
Set-Content -Path (Join-Path $latexSkillDir "SKILL.md") -Value $latexSkill -Encoding UTF8

$socraticSkill = @'
---
name: socratic_tutor
description: Teach Hasin a concept interactively using the Socratic method — check prerequisites with questions first, then build up one tiny piece at a time, always asking before explaining.
---

# Socratic Tutor Skill
Never explain what the student already knows. Never overwhelm with what they don't.
1. Assess via targeted questions.
2. Build up only what is missing.
3. Confirm understanding before moving forward.
4. One question at a time, smallest possible examples.
'@
Set-Content -Path (Join-Path $socraticSkillDir "SKILL.md") -Value $socraticSkill -Encoding UTF8

# 4. Write Preview Scripts (start_preview.js and chat_preview.js)
Write-Host "[4/6] Writing preview server scripts..." -ForegroundColor Yellow

$startPreviewJs = @'
const { spawn } = require('child_process');
const path = require('path');

const scriptPath = path.join(__dirname, 'chat_preview.js');
const child = spawn(process.execPath, [scriptPath], {
  detached: true,
  stdio: 'ignore'
});

child.unref();
console.log('Chat preview server spawned in background.');
'@
Set-Content -Path (Join-Path $scratchDir "start_preview.js") -Value $startPreviewJs -Encoding UTF8

# Copy chat_preview.js from local repo if available, or write base implementation
$repoChatPreview = Join-Path $PSScriptRoot "..\docs\antigravity_customization_transfer_guide.md"
$sourceChatPreview = Join-Path $userHome ".gemini\antigravity-ide\scratch\chat_preview.js"
$targetChatPreview = Join-Path $scratchDir "chat_preview.js"

if (Test-Path $sourceChatPreview) {
    Copy-Item -Path $sourceChatPreview -Destination $targetChatPreview -Force
} else {
    Write-Host "Please ensure chat_preview.js is populated into $targetChatPreview from the transfer guide." -ForegroundColor Magenta
}

# 5. Write IDE Configuration (keybindings, tasks, settings)
Write-Host "[5/6] Writing IDE tasks, keybindings, and settings..." -ForegroundColor Yellow

$tasksJson = @'
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start LaTeX Chat Preview Server",
      "type": "shell",
      "command": "node $env:USERPROFILE/.gemini/antigravity-ide/scratch/start_preview.js",
      "presentation": {
        "reveal": "never",
        "panel": "new"
      }
    }
  ]
}
'@
Set-Content -Path (Join-Path $vscodeUserDir "tasks.json") -Value $tasksJson -Encoding UTF8

$keybindingsJson = @'
[
  {
    "key": "ctrl+alt+l",
    "command": "runCommands",
    "args": {
      "commands": [
        {
          "command": "workbench.action.terminal.focus"
        },
        {
          "command": "workbench.action.terminal.sendSequence",
          "args": {
            "text": "$env:VSCODE_PID = $PID; node $env:USERPROFILE/.gemini/antigravity-ide/scratch/chat_preview.js\r"
          }
        },
        {
          "command": "simpleBrowser.show",
          "args": "http://localhost:8085/"
        }
      ]
    }
  }
]
'@
Set-Content -Path (Join-Path $vscodeUserDir "keybindings.json") -Value $keybindingsJson -Encoding UTF8

$settingsJson = @'
{
    "python.languageServer": "Default",
    "files.autoSave": "afterDelay",
    "editor.wordWrap": "on",
    "editor.mouseWheelZoom": true,
    "claudeCode.preferredLocation": "panel",
    "vscode-office.openOutline": false,
    "workbench.editorAssociations": {
        "*.pptx": "muty-pptviewer.preview",
        "*.html": "cweijan.htmlViewer"
    },
    "vscode-office.codeMirrorTheme": "Material Dark"
}
'@
Set-Content -Path (Join-Path $vscodeUserDir "settings.json") -Value $settingsJson -Encoding UTF8

$configJson = @'
{
  "userSettings": {
    "globalPermissionGrants": {
      "allow": [
        "command(git status)"
      ]
    }
  }
}
'@
Set-Content -Path (Join-Path $configDir "config.json") -Value $configJson -Encoding UTF8

$mcpJson = @'
{
    "mcpServers": {
        "stitch": {
            "serverUrl": "https://stitch.googleapis.com/mcp",
            "headers": {
                "X-Goog-Api-Key": "<YOUR_STITCH_API_KEY>"
            }
        }
    }
}
'@
Set-Content -Path (Join-Path $configDir "mcp_config.json") -Value $mcpJson -Encoding UTF8

Write-Host "[6/6] Finalizing installation..." -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  ANTIGRAVITY CUSTOMIZATIONS INSTALLED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Installed components:"
Write-Host "  [OK] ~/.gemini/config/AGENTS.md (Global Rules, Persona & Identity)"
Write-Host "  [OK] ~/.gemini/config/skills/latex_preview/SKILL.md"
Write-Host "  [OK] ~/.gemini/config/skills/socratic_tutor/SKILL.md"
Write-Host "  [OK] ~/.gemini/antigravity-ide/scratch/ (chat_preview.js, start_preview.js)"
Write-Host "  [OK] AppData/Roaming/Antigravity IDE/User/ (keybindings, tasks, settings)"
Write-Host "  [OK] ~/.gemini/config/ (config.json, mcp_config.json)"
Write-Host ""
Write-Host "Press Ctrl + Alt + L in Antigravity to launch the LaTeX Preview server!" -ForegroundColor Cyan
