const fs = require('fs');
const http = require('http');
const url = require('url');
const path = require('path');

process.on('uncaughtException', (err) => {
  console.error('[Preview Server] Uncaught Exception:', err);
});
process.on('unhandledRejection', (reason, promise) => {
  console.error('[Preview Server] Unhandled Rejection:', reason);
});

const os = require('os');

let puppeteer = null;
try {
  puppeteer = require('puppeteer-core');
} catch (e) {
  try {
    puppeteer = require('D:/antigravity-remote/server/node_modules/puppeteer-core');
  } catch (err2) {}
}

const PORT = 8085;

function getActiveLogFiles() {
  const brainDir = path.join(os.homedir(), '.gemini', 'antigravity-ide', 'brain');
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
      console.log('[USER_MESSAGE]: Injected message and pressed Enter successfully.');
      try { await browser.disconnect(); } catch (e) {}
      return true;
    }

    try { await browser.disconnect(); } catch (e) {}
    return false;
  } catch (err) {
    console.log('Puppeteer injection failed details:', err.stack || err.message || err);
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
    
    html, body {
      overflow-x: hidden;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }

    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-thumb {
      background: #3f3f46;
      border-radius: 3px;
    }
    
    header {
      width: 100%;
      background-color: var(--bg-chat);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 0;
      position: fixed;
      top: 0;
      z-index: 10;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    
    header h1 {
      margin: 0;
      font-size: 1.05rem;
      font-weight: 600;
      letter-spacing: 0.5px;
      color: var(--accent);
    }

    .status-badge {
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 12px;
      background: #10b98122;
      color: #10b981;
      border: 1px solid #10b98144;
    }
    
    #chat-wrapper {
      width: 100%;
      max-width: 850px;
      margin-top: 60px;
      padding: 20px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding-bottom: 110px;
    }
    
    .message-container {
      display: flex;
      flex-direction: column;
      max-width: 90%;
    }
    
    .message-container.user {
      align-self: flex-end;
      align-items: flex-end;
    }
    
    .message-container.assistant {
      align-self: flex-start;
      align-items: flex-start;
    }
    
    .sender-tag {
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 5px;
      margin-left: 8px;
      margin-right: 8px;
    }
    
    .bubble {
      padding: 14px 18px;
      border-radius: 18px;
      line-height: 1.6;
      font-size: 0.95rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      word-wrap: break-word;
    }
    
    .user .bubble {
      background-color: var(--user-bubble);
      color: #f1f5f9;
      border-bottom-right-radius: 4px;
    }
    
    .assistant .bubble {
      background-color: var(--assistant-bubble);
      color: var(--text-main);
      border-bottom-left-radius: 4px;
      border: 1px solid var(--border-color);
    }
    
    .bubble p {
      margin: 0 0 10px 0;
    }
    .bubble p:last-child {
      margin-bottom: 0;
    }
    
    .bubble code {
      font-family: Consolas, Monaco, "Andale Mono", monospace;
      background-color: rgba(0, 0, 0, 0.3);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.85rem;
    }
    
    .bubble pre {
      background-color: rgba(0, 0, 0, 0.4);
      padding: 12px;
      border-radius: 8px;
      overflow-x: auto;
      margin: 10px 0;
      border: 1px solid var(--border-color);
    }
    
    .bubble pre code {
      background-color: transparent;
      padding: 0;
      font-size: 0.85rem;
    }
    
    .bubble a {
      color: #60a5fa;
      text-decoration: none;
    }
    
    .bubble a:hover {
      text-decoration: underline;
    }
    
    .bubble table {
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0;
    }
    
    .bubble th, .bubble td {
      border: 1px solid var(--border-color);
      padding: 8px;
      text-align: left;
    }
    
    .bubble th {
      background-color: rgba(255, 255, 255, 0.05);
    }
    
    .katex-display {
      margin: 0.8em 0;
      overflow-x: auto;
      overflow-y: hidden;
    }

    #input-bar {
      position: fixed;
      bottom: 0;
      width: 100%;
      background-color: var(--bg-chat);
      border-top: 1px solid var(--border-color);
      padding: 15px 20px;
      box-sizing: border-box;
      display: flex;
      justify-content: center;
      z-index: 10;
    }

    #input-container {
      width: 100%;
      max-width: 800px;
      display: flex;
      gap: 12px;
    }

    #user-input {
      flex: 1;
      background-color: var(--bg-dark);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 12px 16px;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s;
    }

    #user-input:focus {
      border-color: var(--accent);
    }

    #send-btn {
      background-color: var(--accent);
      color: #ffffff;
      border: none;
      border-radius: 8px;
      padding: 0 20px;
      font-weight: 500;
      font-size: 0.95rem;
      cursor: pointer;
      transition: background-color 0.2s;
    }

    #send-btn:hover {
      background-color: var(--accent-hover);
    }

    #send-btn:disabled, #user-input:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
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
      
      // 1. Display Math
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

      // 2. Inline Math
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
          rendered = katex.renderToString(item.math, {
            displayMode: item.display,
            throwOnError: false
          });
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
        if (!res.ok) throw new Error('Failed to fetch data');
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
      } catch (err) {
        console.error('Update error:', err);
      }
    }

    const input = document.getElementById('user-input');
    const btn = document.getElementById('send-btn');

    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      
      input.value = '';
      input.disabled = true;
      btn.disabled = true;
      
      try {
        const res = await fetch('/send', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text })
        });
        if (!res.ok) throw new Error('Send failed');
      } catch (err) {
        console.error('Send error:', err);
        alert('Failed to send message to agent.');
      } finally {
        input.disabled = false;
        btn.disabled = false;
        input.focus();
        updateChat();
      }
    }

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') sendMessage();
    });
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
    req.on('data', chunk => {
      body += chunk.toString();
    });
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
              if (full && fs.existsSync(full)) {
                fs.appendFileSync(full, lineString, 'utf8');
              }
              if (compact && fs.existsSync(compact)) {
                fs.appendFileSync(compact, lineString, 'utf8');
              }
            }

            res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
            res.end(JSON.stringify({ status: 'ok', injected: false }));
            
            setTimeout(() => {
              process.exit(0);
            }, 150);
          }
        } else {
          res.writeHead(400, { 'Content-Type': 'text/plain' });
          res.end('Missing message');
        }
      } catch (e) {
        console.error('Error in /send endpoint:', e);
        res.writeHead(400, { 'Content-Type': 'text/plain' });
        res.end('Invalid JSON or Error');
      }
    });
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not Found');
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log('LaTeX chat preview server is running at http://127.0.0.1:' + PORT + '/');
});
