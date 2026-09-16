# Antigravity IDE Customization Replication Guide

> **Target Audience:** Any Antigravity Agent or Developer setting up a new/secondary machine (e.g., Research PC, lab workstation, secondary laptop) to reproduce Hasin Ishrak's exact Antigravity configuration.
> **Scope:** Global rules, personality & personal info, Socratic tutor, LaTeX preview server, hotkeys, auto-start tasks, and workspace memory architecture.

---

## Architecture & File Map

Antigravity loads customizations from two main tiers:
1. **Global Configuration:** Located at `~/.gemini/config` (or `%USERPROFILE%\.gemini\config`) and `%APPDATA%\Antigravity IDE\User`.
2. **Workspace Configuration:** Located in each project root under `memory/` and `.vscode/`.

```
%USERPROFILE%\.gemini\
├── config\
│   ├── AGENTS.md                                   # Global rules, memory protocol, personality, identity
│   ├── config.json                                 # Global permission grants
│   ├── mcp_config.json                             # Registered MCP servers
│   └── skills\
│       ├── latex_preview\
│       │   └── SKILL.md                            # LaTeX Chat Preview Server skill
│       └── socratic_tutor\
│           └── SKILL.md                            # Interactive Socratic tutoring skill
└── antigravity-ide\
    └── scratch\
        ├── chat_preview.js                         # KaTeX + Marked.js server (Port 8085)
        └── start_preview.js                        # Background process spawner

%APPDATA%\Antigravity IDE\User\
├── keybindings.json                                # Ctrl+Alt+L hotkey for LaTeX preview
├── tasks.json                                      # Auto-start preview on folder open
└── settings.json                                   # Auto-save, word wrap, UI configurations

<Workspace Root>\
├── .vscode\
│   └── settings.json                               # LaTeX Workshop build pipeline (pdflatex/biber)
└── memory\
    ├── personality.md                              # Unfiltered, loving roast persona
    ├── personal_info.md                            # Hasin's profile & credentials
    ├── purpose.md                                  # Mission & roadmap
    ├── state.md                                    # Active goals & blockers
    ├── history.md                                  # Prepending chat log
    └── index.md                                    # Complete file & formula index
```

---

## Method 1: 1-Click Automated Setup (PowerShell)

Save the following script as `setup_antigravity_customizations.ps1` and run it in PowerShell on the target machine:

```powershell
# ==============================================================================
# ANTIGRAVITY IDE CUSTOMIZATION AUTO-INSTALLER
# ==============================================================================
$ErrorActionPreference = "Stop"

$userHome = $env:USERPROFILE
$geminiDir = Join-Path $userHome ".gemini"
$configDir = Join-Path $geminiDir "config"
$skillsDir = Join-Path $configDir "skills"
$scratchDir = Join-Path $geminiDir "antigravity-ide\scratch"
$vscodeUserDir = Join-Path $env:APPDATA "Antigravity IDE\User"

Write-Host "Creating Antigravity configuration directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $configDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $skillsDir "latex_preview") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $skillsDir "socratic_tutor") | Out-Null
New-Item -ItemType Directory -Force -Path $scratchDir | Out-Null
New-Item -ItemType Directory -Force -Path $vscodeUserDir | Out-Null

Write-Host "Writing configuration files..." -ForegroundColor Cyan

# 1. config.json
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

# 2. mcp_config.json
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

# 3. tasks.json
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

# 4. keybindings.json
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

# 5. settings.json (merge/set)
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

# 6. start_preview.js
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

Write-Host "[SUCCESS] All base configs and directories initialized!" -ForegroundColor Green
Write-Host "Now populate AGENTS.md, SKILL.md files, and chat_preview.js using the manual blocks below." -ForegroundColor Yellow
```

---

## Method 2: Detailed File-by-File Specifications

### 1. Global Rules & Identity (`~/.gemini/config/AGENTS.md`)
**Destination Path:** `%USERPROFILE%\.gemini\config\AGENTS.md`

```markdown
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
```

---

### 2. Socratic Tutor Skill (`~/.gemini/config/skills/socratic_tutor/SKILL.md`)
**Destination Path:** `%USERPROFILE%\.gemini\config\skills\socratic_tutor\SKILL.md`

```markdown
---
name: socratic_tutor
description: Teach Hasin a concept interactively using the Socratic method — check prerequisites with questions first, then build up one tiny piece at a time, always asking before explaining.
---

# Socratic Tutor Skill

Use this skill whenever Hasin says "teach me X" or "can you check if I know X" or similar.

## Core Philosophy
**Never explain what the student already knows. Never overwhelm with what they don't.**

1. **Assess** what they already know via targeted questions
2. **Build up** only what they're missing, one atomic concept at a time
3. **Confirm** understanding before moving to the next concept
4. **Connect** new concepts to things they already know

## The Teaching Loop
```
REPEAT:
  1. Identify the NEXT atomic concept needed
  2. Ask ONE question to check if Hasin already knows it
  3. If YES -> acknowledge, move on
  4. If NO  -> teach it with the smallest possible example
  5. Ask a follow-up question to confirm understanding
  6. Only then move to the next concept
UNTIL: the full topic is understood
```

## Rules
- **One question at a time.** Never ask two questions in the same message.
- **Smallest possible examples.**
- **Acknowledge correct answers enthusiastically.** Hasin is a perfectionist and needs confidence boosts.
- **Never dump rules as a list** before checking if they're needed.
- **If Hasin says "I don't know"**, teach that specific piece before moving on.
- **If Hasin says "too much / go slower"**, zoom in further into even smaller atoms.
- **NEVER use an unexplained technical term.** Define in plain English or ask first.

## Session Structure
1. **Context First (MANDATORY)**: 2-4 sentence real-world framing. WHY does this topic exist? Where is it used?
2. **Frame the topic**: "To understand X, we need A, B, C. Let's check what you know."
3. **Prerequisite sweep**: Ask one question per prerequisite.
4. **Teach missing pieces**: Using the loop above.
5. **Connect to assignment/quiz**: Conclude by applying to active coursework.
```

---

### 3. LaTeX Preview Server Skill (`~/.gemini/config/skills/latex_preview/SKILL.md`)
**Destination Path:** `%USERPROFILE%\.gemini\config\skills\latex_preview\SKILL.md`

```markdown
---
name: latex_preview_server
description: Manage, run, and troubleshoot the LaTeX Chat Preview Server on port 8085.
---

# LaTeX Chat Preview Server

This skill describes the LaTeX chat preview system designed for Antigravity IDE to support markdown and LaTeX math rendering (since the IDE's built-in chat window does not natively render equations).

## System Architecture

1. **Background Server**:
   - Location: `%USERPROFILE%/.gemini/antigravity-ide/scratch/chat_preview.js`
   - Port: `8085`
   - Purpose: Reads conversation log `transcript_full.jsonl` dynamically from active conversation directory, serves a web page with KaTeX and Marked.js, and handles incoming POST requests to forward user messages back to the IDE.

2. **Puppeteer Integration**:
   - Remote debugging port: `127.0.0.1:9222`.
   - Interaction: Accesses Lexical Editor state (`__lexicalEditor`) inside `workbench.html` and presses the native Enter key.
   - Fallback: Writes to `scratch/user_message.txt` and appends directly to conversation JSONL.

3. **Global IDE Configurations**:
   - **Auto-Start Task**: Configured in `%APPDATA%/Antigravity IDE/User/tasks.json`.
   - **Shortcut**: `Ctrl + Alt + L` mapped to open Simple Browser `http://localhost:8085/` in `keybindings.json`.

## How to Restart the Server
Run in the IDE terminal:
```powershell
node "$env:USERPROFILE/.gemini/antigravity-ide/scratch/chat_preview.js"
```
```

---

### 4. Background LaTeX Preview Server (`~/.gemini/antigravity-ide/scratch/chat_preview.js`)
**Destination Path:** `%USERPROFILE%\.gemini\antigravity-ide\scratch\chat_preview.js`

```javascript
const fs = require('fs');
const http = require('http');
const url = require('url');
const path = require('path');
const os = require('os');

process.on('uncaughtException', (err) => {
  console.error('[Preview Server] Uncaught Exception:', err);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('[Preview Server] Unhandled Rejection:', reason);
});

let puppeteer = null;
try {
  puppeteer = require('puppeteer-core');
} catch (e) {
  // Optional remote fallback path if installed globally or in specific directory
  try {
    puppeteer = require('D:/antigravity-remote/server/node_modules/puppeteer-core');
  } catch (err2) {
    // Puppeteer optional - server works in direct-log mode without it
  }
}

const PORT = 8085;
const USER_HOME = os.homedir().replace(/\\/g, '/');

function getActiveLogFiles() {
  const brainDir = path.join(USER_HOME, '.gemini/antigravity-ide/brain');
  if (!fs.existsSync(brainDir)) return { full: null, compact: null, convoDir: null };
  try {
    const entries = fs.readdirSync(brainDir, { withFileTypes: true });
    let latestDir = null;
    let latestMtime = 0;
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const fullLog = path.join(brainDir, entry.name, '.system_generated', 'logs', 'transcript_full.jsonl');
        if (fs.existsSync(fullLog)) {
          const stats = fs.statSync(fullLog);
          if (stats.mtimeMs > latestMtime) {
            latestMtime = stats.mtimeMs;
            latestDir = path.join(brainDir, entry.name);
          }
        }
      }
    }
    if (latestDir) {
      return {
        full: path.join(latestDir, '.system_generated', 'logs', 'transcript_full.jsonl'),
        compact: path.join(latestDir, '.system_generated', 'logs', 'transcript.jsonl'),
        convoDir: latestDir
      };
    }
  } catch (err) {
    console.error('Error finding active convo:', err);
  }
  return { full: null, compact: null, convoDir: null };
}

function getMessages() {
  try {
    const { full } = getActiveLogFiles();
    if (!full || !fs.existsSync(full)) return [];
    const content = fs.readFileSync(full, 'utf8');
    const lines = content.split('\n');
    const messages = [];
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.source === 'USER_EXPLICIT' && obj.type === 'USER_INPUT') {
          let text = obj.content || '';
          const match = text.match(/<USER_REQUEST>([\s\S]*?)<\/USER_REQUEST>/);
          if (match) {
            text = match[1].trim();
          }
          messages.push({ sender: 'user', text });
        } else if (obj.source === 'MODEL' && obj.type === 'PLANNER_RESPONSE' && obj.content) {
          messages.push({ sender: 'assistant', text: obj.content });
        }
      } catch (e) {
        // ignore malformed lines
      }
    }
    return messages;
  } catch (err) {
    console.error('Error reading log:', err);
    return [];
  }
}

async function sendMessageToIDE(text) {
  if (!puppeteer) return false;
  let browser;
  try {
    browser = await puppeteer.connect({
      browserURL: 'http://127.0.0.1:9222',
      defaultViewport: null
    });
    const pages = await browser.pages();
    const page = pages.find(p => p.url().includes('workbench.html'));
    if (!page) {
      throw new Error('workbench.html not found');
    }
    
    const focused = await page.evaluate((msgText) => {
      const allEditables = Array.from(document.querySelectorAll('div[contenteditable="true"]'));
      const chatInputs = allEditables.filter(el => el.className.includes('max-h-[300px]'));
      if (chatInputs.length === 0) return false;
      const target = chatInputs[chatInputs.length - 1];
      target.focus();

      const editor = target.__lexicalEditor;
      if (!editor) return false;

      const stateJson = {
        root: {
          children: [
            {
              children: [
                {
                  detail: 0,
                  format: 0,
                  mode: "normal",
                  style: "",
                  text: msgText,
                  type: "text",
                  version: 1
                }
              ],
              direction: "ltr",
              format: "",
              indent: 0,
              type: "paragraph",
              version: 1
            }
          ],
          direction: "ltr",
          format: "",
          indent: 0,
          type: "root",
          version: 1
        }
      };

      const state = editor.parseEditorState(stateJson);
      editor.setEditorState(state);
      return true;
    }, text);

    if (focused) {
      await page.focus('div[contenteditable="true"]');
      await page.keyboard.press('Enter');
      try { await browser.disconnect(); } catch (e) {}
      return true;
    }

    try { await browser.disconnect(); } catch (e) {}
    return false;
  } catch (err) {
    if (browser) {
      try { await browser.disconnect(); } catch (e) {}
    }
    return false;
  }
}

const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LaTeX Chat Preview</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/lib/marked.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <style>
    :root {
      --bg-dark: #121214;
      --bg-chat: #1a1a1e;
      --user-bubble: #1f3b68;
      --assistant-bubble: #282830;
      --text-main: #e2e8f0;
      --text-muted: #94a3b8;
      --border-color: #2d2d34;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
    }
    html, body { overflow-x: hidden; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      margin: 0; padding: 0;
      display: flex; flex-direction: column; align-items: center; min-height: 100vh;
    }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
    header {
      width: 100%; background-color: var(--bg-chat); border-bottom: 1px solid var(--border-color);
      padding: 12px 0; position: fixed; top: 0; z-index: 10; text-align: center;
      display: flex; align-items: center; justify-content: center; gap: 10px;
    }
    header h1 { margin: 0; font-size: 1.05rem; font-weight: 600; color: var(--accent); }
    .status-badge {
      font-size: 0.75rem; padding: 2px 8px; border-radius: 12px;
      background: #10b98122; color: #10b981; border: 1px solid #10b98144;
    }
    #chat-wrapper {
      width: 100%; max-width: 850px; margin-top: 60px; padding: 20px;
      box-sizing: border-box; display: flex; flex-direction: column; gap: 20px; padding-bottom: 110px;
    }
    .message-container { display: flex; flex-direction: column; max-width: 90%; }
    .message-container.user { align-self: flex-end; align-items: flex-end; }
    .message-container.assistant { align-self: flex-start; align-items: flex-start; }
    .sender-tag { font-size: 0.75rem; font-weight: 500; color: var(--text-muted); margin: 0 8px 5px; }
    .bubble {
      padding: 14px 18px; border-radius: 18px; line-height: 1.6; font-size: 0.95rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1); word-wrap: break-word;
    }
    .user .bubble { background-color: var(--user-bubble); color: #f1f5f9; border-bottom-right-radius: 4px; }
    .assistant .bubble {
      background-color: var(--assistant-bubble); color: var(--text-main);
      border-bottom-left-radius: 4px; border: 1px solid var(--border-color);
    }
    .bubble p { margin: 0 0 10px 0; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble code {
      font-family: Consolas, Monaco, monospace; background-color: rgba(0, 0, 0, 0.3);
      padding: 2px 6px; border-radius: 4px; font-size: 0.85rem;
    }
    .bubble pre {
      background-color: rgba(0, 0, 0, 0.4); padding: 12px; border-radius: 8px;
      overflow-x: auto; margin: 10px 0; border: 1px solid var(--border-color);
    }
    #input-bar {
      position: fixed; bottom: 0; left: 0; width: 100%; background-color: var(--bg-chat);
      border-top: 1px solid var(--border-color); padding: 15px 0; display: flex; justify-content: center; z-index: 10;
    }
    #input-container { width: 100%; max-width: 850px; display: flex; gap: 10px; padding: 0 20px; box-sizing: border-box; }
    #user-input {
      flex: 1; background-color: #24242a; border: 1px solid var(--border-color);
      border-radius: 8px; color: var(--text-main); padding: 12px 16px; font-size: 0.95rem; outline: none;
    }
    #user-input:focus { border-color: var(--accent); }
    #send-btn {
      background-color: var(--accent); color: #ffffff; border: none; border-radius: 8px;
      padding: 0 20px; font-weight: 500; cursor: pointer;
    }
    #send-btn:hover { background-color: var(--accent-hover); }
  </style>
</head>
<body>
  <header>
    <h1>LaTeX Chat Preview</h1>
    <span class="status-badge">Live</span>
  </header>
  <div id="chat-wrapper">
    <div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 40px;">
      Connecting to conversation stream...
    </div>
  </div>
  <div id="input-bar">
    <div id="input-container">
      <input type="text" id="user-input" placeholder="Type a message for Antigravity..." autocomplete="off">
      <button id="send-btn">Send</button>
    </div>
  </div>
  <script>
    let lastMessageHash = '';
    function computeHash(messages) {
      return messages.map(m => m.text.length + m.sender).join('-');
    }
    function renderMarkdownWithLatex(rawText) {
      const mathBlocks = [];
      let text = rawText || '';
      text = text.replace(/\\$\\$([\\s\\S]*?)\\$\\$/g, (match, math) => {
        const id = 'MATHPLACEHOLDER' + mathBlocks.length + 'END';
        mathBlocks.push({ display: true, math });
        return id;
      });
      text = text.replace(/\\\\\\[([\\s\\S]*?)\\\\\\]/g, (match, math) => {
        const id = 'MATHPLACEHOLDER' + mathBlocks.length + 'END';
        mathBlocks.push({ display: true, math });
        return id;
      });
      text = text.replace(/\\$([^\\$\\n]+?)\\$/g, (match, math) => {
        const id = 'MATHPLACEHOLDER' + mathBlocks.length + 'END';
        mathBlocks.push({ display: false, math });
        return id;
      });
      text = text.replace(/\\\\\\(([\\s\\S]*?)\\\\\\)/g, (match, math) => {
        const id = 'MATHPLACEHOLDER' + mathBlocks.length + 'END';
        mathBlocks.push({ display: false, math });
        return id;
      });
      let html = marked.parse(text);
      mathBlocks.forEach((item, index) => {
        const placeholder = 'MATHPLACEHOLDER' + index + 'END';
        let rendered = '';
        try {
          rendered = katex.renderToString(item.math, { displayMode: item.display, throwOnError: false });
        } catch (e) {
          rendered = item.math;
        }
        html = html.split(placeholder).join(rendered);
      });
      return html;
    }
    async function updateChat() {
      try {
        const res = await fetch('/data');
        if (!res.ok) return;
        const messages = await res.json();
        const currentHash = computeHash(messages);
        if (currentHash === lastMessageHash) return;
        lastMessageHash = currentHash;
        const chatWrapper = document.getElementById('chat-wrapper');
        chatWrapper.innerHTML = '';
        if (messages.length === 0) {
          chatWrapper.innerHTML = '<div style="text-align: center; color: var(--text-muted); font-size: 0.85rem; margin-top: 40px;">No messages found in active conversation.</div>';
          return;
        }
        messages.forEach(msg => {
          const container = document.createElement('div');
          container.className = 'message-container ' + msg.sender;
          const tag = document.createElement('div');
          tag.className = 'sender-tag';
          tag.textContent = msg.sender === 'user' ? 'YOU' : 'ANTIGRAVITY';
          const bubble = document.createElement('div');
          bubble.className = 'bubble';
          bubble.innerHTML = renderMarkdownWithLatex(msg.text);
          container.appendChild(tag);
          container.appendChild(bubble);
          chatWrapper.appendChild(container);
        });
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      } catch (err) {}
    }
    const input = document.getElementById('user-input');
    const btn = document.getElementById('send-btn');
    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      input.value = ''; input.disabled = true; btn.disabled = true;
      try {
        await fetch('/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
      } catch (err) {
      } finally {
        input.disabled = false; btn.disabled = false; input.focus(); updateChat();
      }
    }
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });
    btn.addEventListener('click', sendMessage);
    setInterval(updateChat, 1500);
    updateChat();
  </script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;
  if (pathname === '/' || pathname === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
  } else if (pathname === '/data') {
    res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
    res.end(JSON.stringify(getMessages()));
  } else if (pathname === '/send' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', async () => {
      try {
        const data = JSON.parse(body);
        if (data.message) {
          const success = await sendMessageToIDE(data.message);
          if (success) {
            res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
            res.end(JSON.stringify({ status: 'ok', injected: true }));
          } else {
            const { full, compact, convoDir } = getActiveLogFiles();
            if (convoDir) {
              const messageFilePath = path.join(convoDir, 'scratch', 'user_message.txt');
              try { fs.writeFileSync(messageFilePath, data.message, 'utf8'); } catch(e){}
              const userMsgLine = {
                step_index: Date.now(),
                source: 'USER_EXPLICIT',
                type: 'USER_INPUT',
                status: 'DONE',
                created_at: new Date().toISOString(),
                content: `<USER_REQUEST>\n${data.message}\n</USER_REQUEST>`
              };
              const lineString = JSON.stringify(userMsgLine) + '\n';
              if (full && fs.existsSync(full)) fs.appendFileSync(full, lineString, 'utf8');
              if (compact && fs.existsSync(compact)) fs.appendFileSync(compact, lineString, 'utf8');
            }
            res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
            res.end(JSON.stringify({ status: 'ok', injected: false }));
            setTimeout(() => { process.exit(0); }, 150);
          }
        }
      } catch (e) {
        res.writeHead(400, { 'Content-Type': 'text/plain' });
        res.end('Error');
      }
    });
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not Found');
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`LaTeX chat preview server is running at http://127.0.0.1:${PORT}/`);
});
```

---

### 5. LaTeX Workshop VS Code Recipe (`<workspace>/.vscode/settings.json`)
**Destination Path:** `<project_root>\.vscode\settings.json`

```json
{
    "latex-workshop.latex.recipes": [
        {
            "name": "pdflatex ➞ biber ➞ pdflatex ➞ pdflatex",
            "tools": [
                "pdflatex",
                "biber",
                "pdflatex",
                "pdflatex"
            ]
        }
    ],
    "latex-workshop.latex.tools": [
        {
            "name": "pdflatex",
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOC%"
            ]
        },
        {
            "name": "biber",
            "command": "biber",
            "args": [
                "%DOCFILE%"
            ]
        }
    ],
    "latex-workshop.latex.recipe.default": "first"
}
```

---

## Verification & Sanity Testing

After running the installer or placing the files on the new PC:

1. **Verify Global Rules Loaded:**
   Start any new conversation in Antigravity. Check that the assistant automatically checks for `memory/` and adopts the unfiltered, emoji-rich personality.
2. **Verify Socratic Tutor Skill:**
   Type: `"teach me how PCGrad works"` -> The agent should immediately invoke `socratic_tutor` and ask a prerequisite question instead of dumping an explanation.
3. **Verify LaTeX Chat Preview Server:**
   Press `Ctrl + Alt + L` in the IDE.
   - Terminal launches `chat_preview.js` on port `8085`.
   - Simple Browser tab opens `http://localhost:8085/`.
   - Math equations (e.g. `$$L_{total} = \sum w_i L_i$$`) render crisply with KaTeX!
