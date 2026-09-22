const BACKEND_URL = "https://vidmentor-6ixd.onrender.com"; 

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CHAT") {
    fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: message.video_id,
        query: message.query,
        current_time: message.current_time,
      }),
    })
      .then((res) => res.json())
      .then((data) => sendResponse(data))
      .catch((err) => sendResponse({ success: false, error: String(err) }));

    return true; // keeps sendResponse alive for async fetch
  }

  if (message.type === "INDEX") {
    fetch(`${BACKEND_URL}/index`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: message.video_id }),
    })
      .then((res) => res.json())
      .then((data) => sendResponse(data))
      .catch((err) => sendResponse({ success: false, error: String(err) }));

    return true;
  }
});