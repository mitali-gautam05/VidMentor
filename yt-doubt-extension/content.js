const INITIAL_GREETING = "Hi! Ask me anything about this video and I'll answer using its transcript.";
const REQUEST_TIMEOUT_MS = 25000;
let isSending = false;
const indexedVideos = new Set();

function getVideoId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("v");
}

function triggerIndexing() {
  const videoId = getVideoId();
  if (!videoId || indexedVideos.has(videoId)) return;
  indexedVideos.add(videoId);

  try {
    chrome.runtime.sendMessage({ type: "INDEX", video_id: videoId }, () => {
      if (chrome.runtime.lastError) {
        // no-op
      }
    });
  } catch (err) {
    // Extension context invalidated
  }
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
    if (p && typeof p.catch === "function") p.catch(() => { });
  }
}

function removeWidget() {
  document.querySelectorAll("#yt-doubt-button, #yt-doubt-panel").forEach((el) => el.remove());
  isSending = false;
}

function clearChat(e) {
  if (e) {
    if (typeof e.preventDefault === "function") e.preventDefault();
    if (typeof e.stopPropagation === "function") e.stopPropagation();
  }
  if (isSending) return;
  const panel = e?.currentTarget?.closest?.("#yt-doubt-panel") || document.getElementById("yt-doubt-panel");
  const messagesDiv = panel?.querySelector("#yt-doubt-messages") || document.getElementById("yt-doubt-messages");
  if (!messagesDiv) return;
  messagesDiv.innerHTML = "";
  addMessage(INITIAL_GREETING, "bot", messagesDiv);
}

function closeChatPanel() {
  const panel = document.getElementById("yt-doubt-panel");
  if (!panel?.classList.contains("yt-doubt-open")) return;

  panel.classList.remove("yt-doubt-open");
  panel.inert = true;
  panel.setAttribute("aria-hidden", "true");
  const button = document.getElementById("yt-doubt-button");
  button?.setAttribute("aria-expanded", "false");
  button?.focus();
}

function createChatWidget() {
  if (!document.body) return;
  if (location.pathname !== "/watch") return; // only on video pages
  if (document.getElementById("yt-doubt-button") || document.getElementById("yt-doubt-panel")) return;

  const button = document.createElement("button");
  button.id = "yt-doubt-button";
  button.type = "button";
  button.setAttribute("aria-label", "Open VidMentor chat");
  button.setAttribute("aria-controls", "yt-doubt-panel");
  button.setAttribute("aria-expanded", "false");
  button.textContent = "❓ Ask Doubt";
  document.body.appendChild(button);

  const panel = document.createElement("div");
  panel.id = "yt-doubt-panel";
  panel.inert = true;
  panel.setAttribute("aria-hidden", "true");
  panel.innerHTML = `
    <div id="yt-doubt-header">
      <span>Ask about this video</span>
      <div id="yt-doubt-header-actions">
        <button id="yt-doubt-clear" type="button" aria-label="Clear chat">Clear</button>
        <button id="yt-doubt-close" type="button" aria-label="Close chat">✕</button>
      </div>
    </div>
    <div id="yt-doubt-messages" role="log" aria-label="Chat messages" aria-live="polite"></div>
    <div id="yt-doubt-input-row">
      <input id="yt-doubt-input" type="text" placeholder="Type your doubt..." autocomplete="off" />
      <button id="yt-doubt-send">Send</button>
    </div>
  `;
  document.body.appendChild(panel);

  addMessage(INITIAL_GREETING, "bot");

  button.addEventListener("click", () => {
    if (panel.classList.contains("yt-doubt-open")) {
      closeChatPanel();
      return;
    }

    panel.classList.add("yt-doubt-open");
    panel.inert = false;
    panel.removeAttribute("aria-hidden");
    button.setAttribute("aria-expanded", "true");
    document.getElementById("yt-doubt-input")?.focus();
    triggerIndexing();
  });

  const clearBtn = panel.querySelector("#yt-doubt-clear");
  const closeBtn = panel.querySelector("#yt-doubt-close");
  const sendBtn = panel.querySelector("#yt-doubt-send");
  const input = panel.querySelector("#yt-doubt-input");

  closeBtn?.addEventListener("click", closeChatPanel);
  clearBtn?.addEventListener("click", clearChat);
  sendBtn?.addEventListener("click", sendMessage);

  input.addEventListener("keydown", (e) => {
    // Stop YouTube shortcuts (space, k, f, m, etc.) from firing while typing
    e.stopPropagation();
    if (e.key === "Escape") {
      closeChatPanel();
      return;
    }
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
    closeChatPanel();
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

function addMessage(text, sender, container = null) {
  const messagesDiv = container || document.getElementById("yt-doubt-messages");
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
  document.querySelectorAll("#yt-doubt-input").forEach((input) => {
    input.disabled = !enabled;
  });
  document.querySelectorAll("#yt-doubt-send").forEach((send) => {
    send.disabled = !enabled;
  });
  document.querySelectorAll("#yt-doubt-clear").forEach((clear) => {
    clear.disabled = !enabled;
  });
}

function finishSending() {
  isSending = false;
  removeTypingIndicator();
  setInputEnabled(true);
  if (document.getElementById("yt-doubt-panel")?.classList.contains("yt-doubt-open")) {
    document.getElementById("yt-doubt-input")?.focus();
  }
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

  let completed = false;
  let timerId = null;

  const handleDone = (callback) => {
    if (completed) return;
    completed = true;
    if (timerId !== null) {
      clearTimeout(timerId);
      timerId = null;
    }
    try {
      if (getVideoId() === videoId) {
        callback();
      }
    } finally {
      finishSending();
    }
  };

  timerId = setTimeout(() => {
    handleDone(() => {
      addMessage("⚠️ Request timed out. The server took too long to respond. Please try again.", "bot");
    });
  }, REQUEST_TIMEOUT_MS);

  try {
    chrome.runtime.sendMessage(
      { type: "CHAT", video_id: videoId, query, current_time: currentTime },
      (response) => {
        handleDone(() => {
          const lastError = chrome.runtime.lastError; // read it to avoid console warnings
          if (lastError) {
            addMessage("⚠️ Connection to the extension was lost. Please refresh this page and try again.", "bot");
            return;
          }

          if (response && response.success) {
            addMessage(response.answer, "bot");
          } else {
            addMessage(`⚠️ ${response?.error || "Something went wrong. Try again."}`, "bot");
          }
        });
      }
    );
  } catch (err) {
    // Happens when the extension was reloaded while this tab stayed open
    handleDone(() => {
      addMessage("⚠️ Extension was updated. Please refresh this page and try again.", "bot");
    });
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
