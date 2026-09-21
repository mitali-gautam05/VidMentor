let isSending = false;

function getVideoId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("v");
}

function getVideoElement() {
  return document.querySelector("video");
}

function getCurrentTime() {
  const video = getVideoElement();
  return video ? video.currentTime : null;
}

function seekTo(seconds) {
  const video = getVideoElement();
  if (video) {
    video.currentTime = seconds;
    const p = video.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  }
}

function removeWidget() {
  document.getElementById("yt-doubt-button")?.remove();
  document.getElementById("yt-doubt-panel")?.remove();
  isSending = false;
}

function createChatWidget() {
  if (!document.body) return;
  if (location.pathname !== "/watch") return; // only on video pages
  if (document.getElementById("yt-doubt-button")) return;

  const button = document.createElement("div");
  button.id = "yt-doubt-button";
  button.textContent = "❓ Ask Doubt";
  document.body.appendChild(button);

  const panel = document.createElement("div");
  panel.id = "yt-doubt-panel";
  panel.innerHTML = `
    <div id="yt-doubt-header">
      <span>Ask about this video</span>
      <span id="yt-doubt-close">✕</span>
    </div>
    <div id="yt-doubt-messages"></div>
    <div id="yt-doubt-input-row">
      <input id="yt-doubt-input" type="text" placeholder="Type your doubt..." autocomplete="off" />
      <button id="yt-doubt-send">Send</button>
    </div>
  `;
  document.body.appendChild(panel);

  addMessage("Hi! Ask me anything about this video and I'll answer using its transcript.", "bot");

  button.addEventListener("click", () => {
    panel.classList.toggle("yt-doubt-open");
    if (panel.classList.contains("yt-doubt-open")) {
      document.getElementById("yt-doubt-input")?.focus();
    }
  });

  document.getElementById("yt-doubt-close").addEventListener("click", () => {
    panel.classList.remove("yt-doubt-open");
  });

  const input = document.getElementById("yt-doubt-input");
  document.getElementById("yt-doubt-send").addEventListener("click", sendMessage);

  input.addEventListener("keydown", (e) => {
    // Stop YouTube shortcuts (space, k, f, m, etc.) from firing while typing
    e.stopPropagation();
    if (e.key === "Enter" && !e.isComposing) {
      e.preventDefault();
      sendMessage();
    }
  });
  input.addEventListener("keyup", (e) => e.stopPropagation());
  input.addEventListener("keypress", (e) => e.stopPropagation());
}

// Registered once (not on every widget re-creation)
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    document.getElementById("yt-doubt-panel")?.classList.remove("yt-doubt-open");
  }
});

function formatAnswerWithTimestampLinks(text) {
  // 1. Escape HTML (bot output is inserted with innerHTML)
  let safe = String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // 2. Markdown bullets "- item" / "* item" -> "• item"
  safe = safe.replace(/^[ \t]*[-*][ \t]+/gm, "• ");

  // 3. **bold** -> <strong>
  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // 4. Timestamps (m:ss, mm:ss or h:mm:ss) -> clickable
  return safe.replace(/\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b/g, (match, h, m, s) => {
    const totalSeconds =
      (h ? parseInt(h, 10) * 3600 : 0) + parseInt(m, 10) * 60 + parseInt(s, 10);
    return `<span class="yt-doubt-timestamp-link" data-seconds="${totalSeconds}">${match}</span>`;
  });
}

function addMessage(text, sender) {
  const messagesDiv = document.getElementById("yt-doubt-messages");
  if (!messagesDiv) return null;

  const msg = document.createElement("div");
  msg.className = `yt-doubt-msg yt-doubt-${sender}`;

  if (sender === "bot") {
    msg.innerHTML = formatAnswerWithTimestampLinks(text);
    msg.querySelectorAll(".yt-doubt-timestamp-link").forEach((el) => {
      el.addEventListener("click", () => seekTo(parseInt(el.dataset.seconds, 10)));
    });
  } else {
    msg.textContent = text;
  }

  messagesDiv.appendChild(msg);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return msg;
}

function showTypingIndicator() {
  const messagesDiv = document.getElementById("yt-doubt-messages");
  if (!messagesDiv) return;

  const typing = document.createElement("div");
  typing.className = "yt-doubt-typing";
  typing.id = "yt-doubt-typing-indicator";
  typing.innerHTML = "<span></span><span></span><span></span>";
  messagesDiv.appendChild(typing);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById("yt-doubt-typing-indicator");
  if (el) el.remove();
}

function setInputEnabled(enabled) {
  const input = document.getElementById("yt-doubt-input");
  const send = document.getElementById("yt-doubt-send");
  if (input) input.disabled = !enabled;
  if (send) send.disabled = !enabled;
}

function finishSending() {
  isSending = false;
  removeTypingIndicator();
  setInputEnabled(true);
  document.getElementById("yt-doubt-input")?.focus();
}

function sendMessage() {
  if (isSending) return;

  const input = document.getElementById("yt-doubt-input");
  if (!input) return;

  const query = input.value.trim();
  if (!query) return;

  const videoId = getVideoId();
  const currentTime = getCurrentTime();

  addMessage(query, "user");
  input.value = "";
  isSending = true;
  setInputEnabled(false);
  showTypingIndicator();

  try {
    chrome.runtime.sendMessage(
      { type: "CHAT", video_id: videoId, query, current_time: currentTime },
      (response) => {
        const lastError = chrome.runtime.lastError; // read it to avoid console warnings
        finishSending();

        if (lastError) {
          addMessage("⚠️ Connection to the extension was lost. Please refresh this page and try again.", "bot");
          return;
        }

        if (response && response.success) {
          addMessage(response.answer, "bot");
        } else {
          addMessage(`⚠️ ${response?.error || "Something went wrong. Try again."}`, "bot");
        }
      }
    );
  } catch (err) {
    // Happens when the extension was reloaded while this tab stayed open
    finishSending();
    addMessage("⚠️ Extension was updated. Please refresh this page and try again.", "bot");
  }
}

function init() {
  createChatWidget();

  let lastUrl = location.href;
  new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      // Remove old widget so a fresh one binds to the new video
      removeWidget();
      createChatWidget();
    }
  }).observe(document.body, { childList: true, subtree: true });
}

if (document.body) {
  init();
} else {
  document.addEventListener("DOMContentLoaded", init, { once: true });
}