const DEFAULT_BACKEND_URL = "https://vidmentor-6ixd.onrender.com";

const indexTasks = new Map();
const transcriptCache = new Map();
const transcriptFetchPromises = new Map();

async function getBackendUrl() {
  try {
    const stored = await chrome.storage.local.get("backend_url");
    return stored.backend_url || DEFAULT_BACKEND_URL;
  } catch {
    return DEFAULT_BACKEND_URL;
  }
}

async function parseResponse(res) {
  try {
    const data = await res.json();
    if (!res.ok) {
      // FastAPI 422 responses carry detail as an array of objects, which
      // would render as "[object Object]" in the widget; only pass strings
      // through, falling back to a generic message otherwise
      const err =
        typeof data?.error === "string"
          ? data.error
          : typeof data?.detail === "string"
            ? data.detail
            : `Server error (${res.status})`;
      return { success: false, error: err };
    }
    return data;
  } catch (err) {
    return {
      success: false,
      error: `Server returned invalid response (status ${res.status})`,
    };
  }
}

// One random id per browser, created on first use and kept in chrome.storage.local
// so the backend can keep chat history separate for each user.
let sessionIdPromise = null;

function getSessionId() {
  if (!sessionIdPromise) {
    sessionIdPromise = chrome.storage.local
      .get("session_id")
      .then(async (stored) => {
        if (stored.session_id) return stored.session_id;
        const id = crypto.randomUUID();
        await chrome.storage.local.set({ session_id: id });
        return id;
      })
      .catch(() => crypto.randomUUID()); // storage unavailable: per-worker id
  }
  return sessionIdPromise;
}

/**
 * Fetches transcript segments directly from YouTube using the browser's context.
 * Bypasses cloud datacenter IP blocks on Render by obtaining caption tracks via the InnerTube API.
 */
async function fetchBrowserTranscript(videoId) {
  if (!videoId) return null;
  if (transcriptCache.has(videoId)) {
    return transcriptCache.get(videoId);
  }
  if (transcriptFetchPromises.has(videoId)) {
    return transcriptFetchPromises.get(videoId);
  }

  const promise = (async () => {
    try {
      // 1. Fetch YouTube watch page HTML to obtain INNERTUBE_API_KEY
      const watchRes = await fetch(`https://www.youtube.com/watch?v=${videoId}`, {
        credentials: "omit",
      });
      if (!watchRes.ok) return null;
      const html = await watchRes.text();

      const keyMatch = html.match(/"INNERTUBE_API_KEY":\s*"([a-zA-Z0-9_-]+)"/);
      if (!keyMatch) return null;
      const apiKey = keyMatch[1];

      // 2. Call InnerTube player API with ANDROID client context (signed, unblocked timedtext URLs)
      const playerRes = await fetch(`https://www.youtube.com/youtubei/v1/player?key=${apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          context: {
            client: {
              clientName: "ANDROID",
              clientVersion: "20.10.38",
            },
          },
          videoId: videoId,
        }),
      });
      if (!playerRes.ok) return null;
      const playerData = await playerRes.json();
      const captionTracks = playerData?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
      if (!Array.isArray(captionTracks) || captionTracks.length === 0) {
        return [];
      }

      // 3. Select preferred caption track (English, Hindi, or first available)
      const selectedTrack =
        captionTracks.find((t) => t.languageCode === "en") ||
        captionTracks.find((t) => t.languageCode === "hi") ||
        captionTracks[0];

      if (!selectedTrack?.baseUrl) return [];

      // 4. Fetch the timedtext XML from the signed baseUrl
      const capRes = await fetch(selectedTrack.baseUrl);
      if (!capRes.ok) return [];
      const xml = await capRes.text();

      // 5. Parse <p t="startMs" d="durMs">text</p> into standardized segments
      const segments = [];
      const pRegex = /<p\s+t="(\d+)"\s+d="(\d+)"[^>]*>([\s\S]*?)<\/p>/g;
      let match;
      while ((match = pRegex.exec(xml)) !== null) {
        const startMs = parseInt(match[1], 10);
        const durMs = parseInt(match[2], 10);
        const cleanText = match[3]
          .replace(/&amp;/g, "&")
          .replace(/&lt;/g, "<")
          .replace(/&gt;/g, ">")
          .replace(/&quot;/g, '"')
          .replace(/&#39;/g, "'")
          .replace(/<[^>]+>/g, "")
          .trim();

        if (cleanText) {
          segments.push({
            text: cleanText,
            start: startMs / 1000.0,
            duration: durMs / 1000.0,
          });
        }
      }

      transcriptCache.set(videoId, segments);
      return segments;
    } catch (err) {
      console.warn("Browser transcript fetch failed, falling back to server:", err);
      return null;
    } finally {
      transcriptFetchPromises.delete(videoId);
    }
  })();

  transcriptFetchPromises.set(videoId, promise);
  return promise;
}

function indexVideo(videoId) {
  if (!videoId) {
    return Promise.resolve({ success: false, error: "Missing video_id" });
  }

  if (indexTasks.has(videoId)) {
    return indexTasks.get(videoId);
  }

  const promise = (async () => {
    try {
      const backendUrl = await getBackendUrl();
      const browserTranscript = await fetchBrowserTranscript(videoId);

      const payload = { video_id: videoId };
      if (Array.isArray(browserTranscript) && browserTranscript.length > 0) {
        payload.transcript = browserTranscript;
      }

      const res = await fetch(`${backendUrl}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await parseResponse(res);
      if (!data.success) {
        indexTasks.delete(videoId);
      }
      return data;
    } catch (err) {
      indexTasks.delete(videoId);
      return { success: false, error: String(err) };
    }
  })();

  indexTasks.set(videoId, promise);
  return promise;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "CHAT") {
    (async () => {
      try {
        if (message.video_id && indexTasks.has(message.video_id)) {
          const indexResult = await indexTasks.get(message.video_id);
          if (!indexResult.success) {
            indexTasks.delete(message.video_id);
          }
        }

        const backendUrl = await getBackendUrl();
        const sessionId = await getSessionId();
        const browserTranscript = await fetchBrowserTranscript(message.video_id);

        const chatPayload = {
          video_id: message.video_id,
          query: message.query,
          current_time: message.current_time,
          session_id: sessionId,
        };
        if (Array.isArray(browserTranscript) && browserTranscript.length > 0) {
          chatPayload.transcript = browserTranscript;
        }

        const res = await fetch(`${backendUrl}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(chatPayload),
        });
        const data = await parseResponse(res);
        sendResponse(data);
      } catch (err) {
        sendResponse({ success: false, error: String(err) });
      }
    })();

    return true; // keeps sendResponse alive for async fetch
  }

  if (message.type === "INDEX") {
    indexVideo(message.video_id)
      .then((data) => sendResponse(data))
      .catch((err) => sendResponse({ success: false, error: String(err) }));

    return true;
  }
});
