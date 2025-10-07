/**
 * server.js
 * Node/Express backend for PDF -> Text -> Gemini QG (MCQ, SAQ, LAQ)
 * Updated: batch MCQs (1 call for 4 MCQs) + parallel SAQ/LAQ (limited concurrency)
 * LAQs now reliably generate 3 questions with fallback if model output is empty.
 */

const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { GoogleGenerativeAI } = require("@google/generative-ai");

if (typeof global.DOMMatrix === "undefined") {
  global.DOMMatrix = class DOMMatrix {
    constructor() {}
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
  global.Path2D = class Path2D { constructor() {} };
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
  dest: path.join(process.cwd(), "tmp"),
  limits: { fileSize: 50 * 1024 * 1024 }
});

const JOBS = {};

/* PDF extraction helper */
async function extractTextFromPdf(filePath) {
  const dataBuffer = fs.readFileSync(filePath);
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

/* Gemini call */
async function callGeminiModel(prompt) {
  try {
    const model = genAI.getGenerativeModel({ model: "models/gemini-2.5-flash" });
    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text().trim();
  } catch (err) {
    console.error("Gemini API error:", err);
    throw err;
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
app.get("/", (req, res) => res.send("PDF QG backend (multi-question mode)"));

app.post("/api/upload", upload.single("file"), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: "No file uploaded" });
  if (!file.mimetype?.includes("pdf")) {
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
    return res.status(400).json({ error: "File not PDF" });
  }

  const id = uuidv4();
  JOBS[id] = { id, status: "processing", createdAt: Date.now(), questions: [] };
  const TARGET = { mcq: 4, saq: 3, laq: 3 };

  try {
    const { text } = await extractTextFromPdf(file.path);
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
        JOBS[id].questions.push({ id: `${id}-mcq-${i}`, type: "mcq", error: err?.message || "MCQ generation failed" });
      }
    }

    // --- 2) SAQs ---
    const saqTasks = new Array(TARGET.saq).fill(0).map((_, idx) => async () => {
      const chunk = chunks.length > 0 ? chunks[idx % chunks.length] : fallbackChunk;
      return await callModelByType("saq", chunk);
    });
    const saqResults = await runWithConcurrency(saqTasks, 3);
    saqResults.forEach((r, idx) => JOBS[id].questions.push({ id: `${id}-saq-${idx}`, type: "saq", ...r }));

    // --- 3) LAQs ---
    const laqTasks = new Array(TARGET.laq).fill(0).map((_, idx) => async () => {
      const chunk = chunks.length > 0 ? chunks[idx % chunks.length] : fallbackChunk;
      const laq = await callModelByType("laq", chunk);
      return { id: `${id}-laq-${idx}`, type: "laq", question: laq.question, answer: laq.points, rawModelOutput: laq.rawModelOutput };
    });
    const laqResults = await runWithConcurrency(laqTasks, 2);
    laqResults.forEach(r => JOBS[id].questions.push(r));

    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
    JOBS[id].status = "completed";
    return res.json({ jobId: id, status: "completed", questions: JOBS[id].questions });

  } catch (err) {
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
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

app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
