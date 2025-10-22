/**
 * server.js
 * Node/Express backend for PDF -> Text -> Gemini QG (MCQ, SAQ, LAQ)
 * Updated: batch MCQs (1 call for 4 MCQs) + parallel SAQ/LAQ (limited concurrency)
 * LAQs now reliably generate 3 questions with fallback if model output is empty.
 *
 * Added: SSE streaming chat proxy endpoints:
 *  - GET  /api/chat/stream/:chatId   (SSE client connects here)
 *  - POST /api/chat/send             (start a model request; returns assistantMessageId)
 *
 * Behavior:
 *  - If env.MODEL_STREAM_URL is set, server will stream from that URL and proxy token chunks to SSE clients.
 *  - Otherwise it will fallback to callGeminiModel() and send the full text as a single chunk.
 */

const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { GoogleGenerativeAI } = require("@google/generative-ai");

if (typeof global.DOMMatrix === "undefined") {
  global.DOMMatrix = class DOMMatrix {
    constructor() { }
    multiply() { return this }
    translate() { return this }
    scale() { return this }
    toFloat32Array() { return new Float32Array(6) }
  };
}
if (typeof global.ImageData === "undefined") {
  global.ImageData = class ImageData {
    constructor(data, w, h) { this.data = data; this.width = w; this.height = h }
  };
}
if (typeof global.Path2D === "undefined") {
  global.Path2D = class Path2D { constructor() { } };
}

const pdf = require("pdf-parse");
const cors = require("cors");
const dotenv = require("dotenv");
const { v4: uuidv4 } = require("uuid");
dotenv.config();

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error("Missing GEMINI_API_KEY in environment. See .env.example");
  process.exit(1);
}
const genAI = new GoogleGenerativeAI(API_KEY);

const PORT = process.env.PORT || 5000;
const app = express();
app.use(express.json());
app.use(cors());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }
});

const JOBS = {};

/* PDF extraction helper */
async function extractTextFromPdf(dataBuffer) {
  const pages = [];
  function renderPage(pageData) {
    return pageData.getTextContent().then((textContent) => {
      const text = textContent.items.map((s) => s.str).join(" ");
      pages.push(text.trim());
      return text + "\n\n";
    });
  }
  const data = await pdf(dataBuffer, { pagerender: renderPage });
  return { text: data.text, pages, metadata: { numpages: data.numpages } };
}

/* chunkText helper */
function chunkText(text, maxChars = 3000) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = start + maxChars;
    if (end >= text.length) {
      chunks.push(text.slice(start).trim());
      break;
    }
    const slice = text.slice(start, end);
    const lastNewline = slice.lastIndexOf("\n");
    const lastSpace = slice.lastIndexOf(" ");
    const breakIndex = Math.max(lastNewline, lastSpace);
    if (breakIndex > Math.floor(maxChars * 0.7)) {
      chunks.push(slice.slice(0, breakIndex).trim());
      start += breakIndex;
    } else {
      chunks.push(slice.trim());
      start = end;
    }
  }
  return chunks;
}

/* Prompts */
function mcqPromptFromChunk(chunk) {
  return `Generate exactly 4 distinct Multiple Choice Questions (MCQs) from the passage below. Each MCQ must have 4 options (A, B, C, D) and specify the correct answer. Number them 1..4 and label options with A) B) C) D). Use this exact format for each:

Question <n>: <question_text>
A) <option A>
B) <option B>
C) <option C>
D) <option D>
Answer: <A|B|C|D>

Passage:
${chunk}

Return only the 4 MCQs in the format above (no extra commentary).`;
}

function laqPromptFromChunk(chunk) {
  return `Read the passage below and generate ONE meaningful long-answer/essay question.
- Question must be answerable from the passage.
- Also provide 3-5 key points that a correct answer should include.
- Ensure the question text is not empty.

Passage:
${chunk}

Format:
Question: <Long answer question>
Expected points:
- <Point 1>
- <Point 2>
- <Point 3>`;
}

/* --- parse functions --- */
function parseMcqOutput(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const result = { question: null, options: [], answer: null, raw: text };

  for (const line of lines) {
    if (/^[-_]{2,}$/.test(line)) continue;
    const matchOpt = line.match(/^([A-D])[\)\.\-]\s*(.+)$/i);
    if (matchOpt) { result.options.push(matchOpt[2].trim()); continue; }
    const ansMatch = line.match(/^Answer[:\s\-]*([A-D])/i);
    if (ansMatch) { result.answer = ansMatch[1].toUpperCase(); continue; }
    const qLine = line.replace(/^(?:Question\s*\d*[:\.\s-]*|Q\s*\d*[:\.\s-]*|\d+\.\s*)/i, "").trim();
    if (!result.question) { result.question = qLine; continue; }
  }

  if (result.answer && result.options.length === 4) {
    const idx = result.answer.charCodeAt(0) - 65;
    if (idx >= 0 && idx < result.options.length) result.correct = result.options[idx];
  }

  if (!result.question && result.raw) {
    const splitByOpt = result.raw.split(/(?:\nA[\)\.])/i);
    if (splitByOpt && splitByOpt.length > 0) {
      const candidate = splitByOpt[0].replace(/^(?:Question\s*\d*[:\.\s-]*|Q\s*\d*[:\.\s-]*)/i, "").trim();
      if (candidate) result.question = candidate;
    }
  }

  if (!result.question) result.question = null;
  return result;
}

function parseLaqOutput(raw) {
  const lines = (raw || "").split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  let questionLine = lines.find(l => l.toLowerCase().startsWith("question:")) || "";
  let question = questionLine.replace(/^Question:\s*/i, "");
  if (!question) question = "Write a detailed answer based on the passage above.";
  const points = lines.filter(l => l.startsWith("-")).map(l => l.substring(1).trim());
  return { question, points, rawModelOutput: raw };
}

/**
 * Retries the Gemini API call with exponential backoff on transient errors (503 or 429).
 */
async function callGeminiModel(prompt, maxRetries = 5, initialDelay = 1000) {
  const model = genAI.getGenerativeModel({ model: "models/gemini-2.5-flash" });
  let delay = initialDelay;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      // 1. Wait before retrying (only on attempts 1 through maxRetries-1)
      if (attempt > 0) {
        console.warn(`Transient error detected. Retrying in ${delay / 1000}s (Attempt ${attempt + 1}/${maxRetries}).`);
        await new Promise(resolve => setTimeout(resolve, delay));
        // Exponential backoff: double the delay for the next attempt
        delay *= 2; 
      }
      
      // 2. Make the API call
      const result = await model.generateContent(prompt);
      const response = await result.response;
      return response.text().trim(); // Success: return result

    } catch (err) {
      // 3. Check for retriable errors (503 Service Unavailable, 429 Rate Limit)
      // The error object from the Google Generative AI SDK contains a 'status' property.
      const isRetriable = err.status === 503 || err.status === 429;

      if (attempt < maxRetries - 1 && isRetriable) {
        // If it's retriable and not the last attempt, the loop will continue
        // and trigger the wait logic in the next iteration.
        continue; 
      } else {
        // Log final error and re-throw if not retriable or max retries reached
        console.error("Gemini API error after all retries or non-retriable error:", err);
        throw err;
      }
    }
  }
}

async function callModelByType(type, chunk) {
  if (type === "saq") {
    const prompt = `Generate a short-answer question and answer from the following text:\n\n${chunk}`;
    const raw = await callGeminiModel(prompt);
    const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const question = lines[0] || "SAQ Question not found";
    const answer = lines.slice(1).join(" ") || "Answer not found";
    return { question, answer, raw };
  } else if (type === "mcq") {
    const raw = await callGeminiModel(mcqPromptFromChunk(chunk));
    return { raw };
  } else if (type === "laq") {
    const raw = await callGeminiModel(laqPromptFromChunk(chunk));
    return parseLaqOutput(raw);
  }
  throw new Error(`Unknown type: ${type}`);
}

/* generateMultipleMcqs */
async function generateMultipleMcqs(chunk, n = 4) {
  const raw = await callGeminiModel(mcqPromptFromChunk(chunk));
  let blocks = raw.split(/(?:Question\s*\d+[:\.])|(?:Question\s*:)/i).map(s => s.trim()).filter(Boolean);
  if (!blocks.length) blocks = raw.split(/\n{2,}/).map(s => s.trim()).filter(Boolean);
  const items = [];
  for (let i = 0; i < Math.min(n, blocks.length); i++) {
    const parsed = parseMcqOutput(blocks[i]);
    parsed.raw = blocks[i];
    items.push(parsed);
  }
  while (items.length < n) items.push({ question: null, options: [], answer: null, raw });
  return items.slice(0, n);
}

/* --- limited concurrency runner --- */
async function runWithConcurrency(tasks, concurrency = 3) {
  const results = [];
  const queue = tasks.slice();
  const workers = new Array(Math.max(1, concurrency)).fill(0).map(async () => {
    while (true) {
      const task = queue.shift();
      if (!task) break;
      try { const r = await task(); results.push(r); } catch (e) { results.push({ error: e?.message || String(e) }); }
    }
  });
  await Promise.all(workers);
  return results;
}

/* Routes */
app.get("/", (req, res) => res.send("Welcome to Quest Genie Backend"));

app.post("/api/upload", upload.single("file"), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: "No file uploaded" });
  if (!file.mimetype?.includes("pdf")) {
    return res.status(400).json({ error: "File not PDF" });
  }

  const id = uuidv4();
  JOBS[id] = { id, status: "processing", createdAt: Date.now(), questions: [] };
  const TARGET = { mcq: 4, saq: 3, laq: 3 };

  try {
    const { text } = await extractTextFromPdf(file.buffer);
    const cleaned = (text || "").replace(/\s+\n/g, "\n").replace(/\u00A0/g, " ").trim();
    const chunks = chunkText(cleaned, 2500);
    const fallbackChunk = chunks.length > 0 ? chunks[0] : cleaned || "No content extracted";

    // --- 1) MCQs ---
    let mcqItems = [];
    try {
      mcqItems = await generateMultipleMcqs(fallbackChunk, TARGET.mcq);
      mcqItems.forEach((parsed, idx) => {
        JOBS[id].questions.push({
          id: `${id}-mcq-${idx}`,
          type: "mcq",
          question: parsed.question || "MCQ question not found",
          options: parsed.options,
          answer: parsed.correct || parsed.answer || null,
          rawModelOutput: parsed.raw || ""
        });
      });
    } catch (err) {
      for (let i = 0; i < TARGET.mcq; i++) {
        JOBS[id].questions.push({ id: `${id}-mcq-${i}`, type: "mcq", question: "MCQ generation failed", error: err?.message || "MCQ generation failed" });
      }
    }

    // --- 2) SAQs ---
    const saqTasks = new Array(TARGET.saq).fill(0).map((_, idx) => async () => {
      const chunk = chunks.length > 0 ? chunks[idx % chunks.length] : fallbackChunk;
      try {
        return await callModelByType("saq", chunk);
      } catch (e) {
        return { question: "SAQ generation failed", answer: "SAQ generation failed", error: e?.message || "SAQ generation failed" };
      }
    });
    const saqResults = await runWithConcurrency(saqTasks, 3);
    saqResults.forEach((r, idx) => JOBS[id].questions.push({ id: `${id}-saq-${idx}`, type: "saq", ...r }));

    // --- 3) LAQs ---
    const laqTasks = new Array(TARGET.laq).fill(0).map((_, idx) => async () => {
      const chunk = chunks.length > 0 ? chunks[idx % chunks.length] : fallbackChunk;
      try {
        const laq = await callModelByType("laq", chunk);
        return { id: `${id}-laq-${idx}`, type: "laq", question: laq.question, answer: laq.points, rawModelOutput: laq.rawModelOutput };
      } catch (e) {
        return { id: `${id}-laq-${idx}`, type: "laq", question: "LAQ generation failed", answer: [], error: e?.message || "LAQ generation failed" };
      }
    });
    const laqResults = await runWithConcurrency(laqTasks, 2);
    laqResults.forEach(r => JOBS[id].questions.push(r));

    JOBS[id].status = "completed";
    return res.json({ jobId: id, status: "completed", questions: JOBS[id].questions });

  } catch (err) {
    JOBS[id].status = "failed";
    JOBS[id].error = err.message || String(err);
    return res.status(500).json({ error: err.message || "Processing failed" });
  }
});

app.get("/api/result/:jobId", (req, res) => {
  const job = JOBS[req.params.jobId];
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

app.post("/api/youtube-suggestions", upload.single("file"), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: "No file uploaded" });
  if (!file.mimetype?.includes("pdf")) {
    return res.status(400).json({ error: "File not PDF" });
  }

  const YT_KEY = process.env.GEMINI_YOUTUBE_API_KEY || null; // optional

  // Helper: search YouTube for a query and return first usable video or null
  async function findAvailableYoutubeVideoForQuery(query) {
    if (!YT_KEY) return null;
    try {
      const searchUrl = `https://www.googleapis.com/youtube/v3/search?part=snippet&type=video&maxResults=5&q=${encodeURIComponent(query)}&key=${YT_KEY}`;
      const searchResp = await fetch(searchUrl);
      if (!searchResp.ok) return null;
      const searchJson = await searchResp.json();
      const ids = (searchJson.items || []).map(i => i.id?.videoId).filter(Boolean);
      if (!ids.length) return null;

      // Check video status (privacy/uploadStatus)
      const vidsUrl = `https://www.googleapis.com/youtube/v3/videos?part=status,contentDetails&id=${ids.join(",")}&key=${YT_KEY}`;
      const vidsResp = await fetch(vidsUrl);
      if (!vidsResp.ok) return null;
      const vidsJson = await vidsResp.json();
      for (const v of vidsJson.items || []) {
        const status = v.status || {};
        // uploadStatus may be "processed", privacyStatus should be "public"
        if (status.privacyStatus === "public" && (status.uploadStatus === undefined || status.uploadStatus === "processed")) {
          return { id: v.id, url: `https://www.youtube.com/watch?v=${v.id}`, status };
        }
      }
      return null;
    } catch (e) {
      console.error("YouTube check error:", e);
      return null;
    }
  }

  try {
    const { text } = await extractTextFromPdf(file.buffer);
    const cleaned = (text || "").replace(/\s+\n/g, "\n").replace(/\u00A0/g, " ").trim();
    const chunk = chunkText(cleaned, 4000)[0] || cleaned;

    if (!chunk) {
      return res.status(400).json({ error: "Could not extract text from PDF." });
    }

    // Prompt now asks for an optional "query" field that we can use to search YouTube reliably.
    const prompt = `Based on the following text from a document, suggest 5 relevant YouTube video topics that would be helpful for studying this material. 
For each topic, provide:
- a concise, searchable title (title)
- a brief one-line description (description)
- a short search query optimized for YouTube (query) that will find the most relevant publicly available tutorial (e.g. "linear regression intuition 10 minutes").

Return the result as a JSON array of objects like:
[
  { "title": "Title here", "description": "One-line", "query": "search query here" },
  ...
]

Text:
"${chunk}"`;

    const modelResponse = await callGeminiModel(prompt);

    // Extract JSON array from model output (robust match)
    const jsonMatch = modelResponse.match(/\[\s*\{[\s\S]*\}\s*\]/);
    if (!jsonMatch) {
      console.warn("Model returned no JSON array; returning a fallback suggestion using simple parsing.");
      throw new Error("No valid JSON array found in the AI model's response.");
    }
    const jsonResponse = jsonMatch[0];

    let suggestions;
    try {
      suggestions = JSON.parse(jsonResponse);
      if (!Array.isArray(suggestions)) throw new Error("Parsed suggestions not an array");
    } catch (e) {
      console.error("Failed to parse JSON from model response:", jsonResponse, e);
      throw new Error("Failed to get valid suggestions from the AI model.");
    }

    // If YT key available, validate each suggestion; otherwise skip validation
    const validated = [];
    for (const s of suggestions) {
      // normalize fields
      const title = (s.title || "").toString();
      const description = (s.description || "").toString();
      const query = (s.query || title).toString();

      if (YT_KEY) {
        const found = await findAvailableYoutubeVideoForQuery(query);
        if (found) {
          validated.push({
            title,
            description,
            query,
            youtube: { id: found.id, url: found.url }
          });
          continue;
        } else {
          // try a second pass with title if query failed and title differs
          if (query !== title) {
            const found2 = await findAvailableYoutubeVideoForQuery(title);
            if (found2) {
              validated.push({
                title,
                description,
                query,
                youtube: { id: found2.id, url: found2.url }
              });
              continue;
            }
          }
          // else: no available video for this suggestion -> drop it
          console.log(`No available YouTube video found for suggestion: "${title}" (query: "${query}")`);
        }
      } else {
        // No YT key: keep suggestion but include the query field to help client-side search/validation
        validated.push({ title, description, query });
      }
    }

    // If validation removed all suggestions and we have original suggestions, fallback to returning originals
    const finalVideos = (validated.length > 0) ? validated : suggestions.map(s => ({
      title: s.title || "",
      description: s.description || ""
    }));

    return res.json({ videos: finalVideos });
  } catch (err) {
    console.error("Error getting YouTube suggestions:", err);
    res.status(500).json({ error: err.message || "Failed to get YouTube suggestions." });
  }
});

/* =========================
   SSE / Chat streaming code
   ========================= */

// Map: chatId => [{ id: clientId, res: httpResponse }]
const sseClients = new Map();

// Map: chatId => [{ event, data }]
const bufferedResponses = new Map();

// Helper to send SSE event
function sendSSE(res, event, data) {
  try {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  } catch (e) {
    // client might have disconnected
    console.error('sendSSE error', e);
  }
}

// SSE endpoint for clients to receive streaming model output for a chat
app.get('/api/chat/stream/:chatId', (req, res) => {
  const { chatId } = req.params;
  // SSE headers
  res.set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
  res.flushHeaders();

  // initial hello
  res.write('event: connected\n');
  res.write(`data: ${JSON.stringify({ ok: true })}\n\n`);

  const clientId = uuidv4();
  const entry = { id: clientId, res };
  if (!sseClients.has(chatId)) sseClients.set(chatId, []);
  sseClients.get(chatId).push(entry);

  // If there's a buffered response for this chat, flush it
  if (bufferedResponses.has(chatId)) {
    const buffer = bufferedResponses.get(chatId) || [];
    console.log(`Flushing ${buffer.length} buffered events for chat ${chatId}`);
    for (const { event, data } of buffer) {
      sendSSE(res, event, data);
    }
    // Clear the buffer after flushing
    bufferedResponses.delete(chatId);
  }

  // heartbeat: keep connection alive (optional)
  const pingInterval = setInterval(() => {
    try {
      res.write('event: ping\n');
      res.write(`data: ${JSON.stringify({ ts: Date.now() })}\n\n`);
    } catch (e) { }
  }, 25000);

  req.on('close', () => {
    clearInterval(pingInterval);
    const arr = sseClients.get(chatId) || [];
    sseClients.set(chatId, arr.filter((c) => c.id !== clientId));
  });
});

// POST endpoint: accept user message, return assistantMessageId and start model streaming
app.post('/api/chat/send', async (req, res) => {
  const { chatId, message, metadata } = req.body || {};
  if (!chatId || !message) return res.status(400).json({ error: 'chatId and message required' });

  const assistantMessageId = uuidv4();

  // respond immediately so frontend can optimistic-update
  res.json({ ok: true, assistantMessageId });

  // start streaming to all SSE clients connected to this chatId
  try {
    await streamFromModel(message, chatId, assistantMessageId, metadata);
  } catch (err) {
    console.error('streamFromModel error', err);
    const clients = sseClients.get(chatId) || [];
    for (const { res: clientRes } of clients) {
      sendSSE(clientRes, 'error', { assistantMessageId, message: 'Model stream error' });
    }
  }
});

/**
 * streamFromModel
 * - If process.env.MODEL_STREAM_URL is set, this tries to POST to that URL and stream its response (raw chunks) to SSE clients.
 * - Otherwise it falls back to callGeminiModel(prompt) and sends the entire text as a single chunk + done event.
 *
 * NOTE: adapt parsing of chunks depending on the provider (newline-delimited JSON, SSE, raw text).
 */
async function streamFromModel(userMessage, chatId, assistantMessageId, metadata = {}) {
  const clients = sseClients.get(chatId) || [];
  const isBuffering = clients.length === 0;

  if (isBuffering) {
    console.log(`No SSE clients connected for chat ${chatId}, buffering response...`);
    bufferedResponses.set(chatId, []);
  }

  const sendOrBuffer = (event, data) => {
    if (isBuffering) {
      // Don't buffer pings
      if (event === 'ping') return;
      bufferedResponses.get(chatId).push({ event, data });
    } else {
      for (const { res: clientRes } of clients) {
        sendSSE(clientRes, event, data);
      }
    }
  };

  const modelStreamUrl = process.env.MODEL_STREAM_URL; // optional: provider streaming endpoint
  const apiKey = process.env.MODEL_API_KEY || process.env.GEMINI_API_KEY;

  if (modelStreamUrl) {
    // Attempt streaming HTTP proxy
    const controller = new AbortController();
    const signal = controller.signal;
    try {
      const resp = await fetch(modelStreamUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: apiKey ? `Bearer ${apiKey}` : undefined,
        },
        body: JSON.stringify({
          model: 'gemini-2.5-pro',
          input: userMessage,
          stream: true,
          metadata,
        }),
        signal,
      });

      if (!resp.ok || !resp.body) {
        const bodyText = await resp.text().catch(() => '');
        throw new Error(`Model stream request failed: ${resp.status} ${bodyText}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let done = false;

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) {
          done = true;
          break;
        }
        const chunkText = decoder.decode(value, { stream: true });
        sendOrBuffer('chunk', { assistantMessageId, text: chunkText });
      }

      sendOrBuffer('done', { assistantMessageId });

      try {
        reader.releaseLock();
      } catch (e) { }
    } catch (err) {
      sendOrBuffer('error', { assistantMessageId, message: err?.message || 'streaming failed' });
      if (!isBuffering) throw err; // Don't throw if we are just buffering
    }
  } else {
    // Fallback: non-streaming Gemini call -> forward as single chunk
    try {
      const prompt = userMessage;
      const text = await callGeminiModel(prompt); // existing helper
      sendOrBuffer('chunk', { assistantMessageId, text });
      sendOrBuffer('done', { assistantMessageId });
    } catch (err) {
      sendOrBuffer('error', { assistantMessageId, message: err?.message || 'model error' });
      if (!isBuffering) throw err; // Don't throw if we are just buffering
    }
  }
}

/* =========================
   End of SSE / Chat streaming code
   ========================= */

app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
