/**
 * server.js
 * Node/Express backend for PDF -> Text -> Gemini QG (MCQ, SAQ, LAQ)
 * Updated: batch MCQs (1 call for 4 MCQs) + parallel SAQ/LAQ (limited concurrency).
 */

const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { GoogleGenerativeAI } = require("@google/generative-ai");
if (typeof global.DOMMatrix === 'undefined') {
  global.DOMMatrix = class DOMMatrix { constructor(){} multiply(){return this} translate(){return this} scale(){return this} toFloat32Array(){return new Float32Array(6)} };
}
if (typeof global.ImageData === 'undefined') {
  global.ImageData = class ImageData { constructor(data,w,h){this.data=data;this.width=w;this.height=h} };
}
if (typeof global.Path2D === 'undefined') {
  global.Path2D = class Path2D { constructor(){} };
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
  return `Generate exactly one long-answer/essay question from the following passage. Also list 3-5 key points that a correct answer should include:

Passage: ${chunk}

Format:
Question: <Long answer question>
Expected points:
- <Point 1>
- <Point 2>
- <Point 3>`;
}

/* --- Improved parseMcqOutput --- */
function parseMcqOutput(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const result = { question: null, options: [], answer: null, raw: text };

  for (const line of lines) {
    // skip lines that are obvious separators
    if (/^[-_]{2,}$/.test(line)) continue;

    // Option lines like "A) option text" or "A. option text"
    const matchOpt = line.match(/^([A-D])[\)\.\-]\s*(.+)$/i);
    if (matchOpt) {
      result.options.push(matchOpt[2].trim());
      continue;
    }

    // Answer line like "Answer: C" or "Answer - C"
    const ansMatch = line.match(/^Answer[:\s\-]*([A-D])/i);
    if (ansMatch) {
      result.answer = ansMatch[1].toUpperCase();
      continue;
    }

    // If the line looks like "Question 1: ..." or "1. ..." remove the prefix
    const qLine = line.replace(/^(?:Question\s*\d*[:\.\s-]*|Q\s*\d*[:\.\s-]*|\d+\.\s*)/i, "").trim();

    // If we haven't seen a question yet, treat this line as the question
    if (!result.question) {
      result.question = qLine;
      continue;
    }

    // If it's none of the above and question already set, ignore or accumulate (not necessary)
  }

  // If answer present and options length is 4, set the correct text
  if (result.answer && result.options.length === 4) {
    const idx = result.answer.charCodeAt(0) - 65;
    if (idx >= 0 && idx < result.options.length) {
      result.correct = result.options[idx];
    }
  }

  // Fallbacks: if question still not set, try to extract before the first option marker
  if (!result.question && result.raw) {
    // attempt to grab everything up to the first "A)" or "A." occurrence
    const splitByOpt = result.raw.split(/(?:\nA[\)\.])/i);
    if (splitByOpt && splitByOpt.length > 0) {
      const candidate = splitByOpt[0].replace(/^(?:Question\s*\d*[:\.\s-]*|Q\s*\d*[:\.\s-]*)/i, "").trim();
      if (candidate) result.question = candidate;
    }
  }

  // Final safety: if still missing, set null (your caller can substitute placeholder)
  if (!result.question) result.question = null;

  return result;
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

/* Unified call by type */
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
    return { raw };
  }
  throw new Error(`Unknown type: ${type}`);
}

/* --- Slightly more robust generateMultipleMcqs (optional) --- */
async function generateMultipleMcqs(chunk, n = 4) {
  const raw = await callGeminiModel(mcqPromptFromChunk(chunk));
  // Try splitting by explicit numbered "Question 1:" or "Question 1." or "Question:" or by blank lines
  let blocks = raw.split(/(?:Question\s*\d+[:\.])|(?:Question\s*:)/i).map(s => s.trim()).filter(Boolean);

  // If that produced nothing, fallback to splitting by double newlines
  if (!blocks.length) {
    blocks = raw.split(/\n{2,}/).map(s => s.trim()).filter(Boolean);
  }

  const items = [];
  for (let i = 0; i < Math.min(n, blocks.length); i++) {
    const parsed = parseMcqOutput(blocks[i]);
    parsed.raw = blocks[i];
    items.push(parsed);
  }

  // Try to salvage additional MCQs by scanning the raw output for option groups
  if (items.length < n) {
    // Find all occurrences that look like a block containing A) ... D)
    const candidateBlocks = raw.split(/(?=\n?[A-D][\)\.]\s+)/).map(s => s.trim()).filter(Boolean);
    for (const block of candidateBlocks) {
      if (items.length >= n) break;
      if (items.some(it => it.raw === block)) continue;
      const parsed = parseMcqOutput(block);
      parsed.raw = block;
      // ensure we at least have options or question
      if (parsed.question || parsed.options.length) items.push(parsed);
    }
  }

  // Pad with placeholders if still fewer than n
  while (items.length < n) {
    items.push({ question: null, options: [], answer: null, raw: raw });
  }

  return items.slice(0, n);
}

/* --- NEW: limited concurrency runner --- */
async function runWithConcurrency(tasks, concurrency = 3) {
  const results = [];
  const queue = tasks.slice();
  const workers = new Array(Math.max(1, concurrency)).fill(0).map(async () => {
    while (true) {
      const task = queue.shift();
      if (!task) break;
      try {
        const r = await task();
        results.push(r);
      } catch (e) {
        results.push({ error: e?.message || String(e) });
      }
    }
  });
  await Promise.all(workers);
  return results;
}

/* Simple root */
app.get("/", (req, res) => res.send("PDF QG backend (multi-question mode)"));

/* POST /api/upload - generate 4 MCQ, 3 SAQ, 3 LAQ and return questions in response */
app.post("/api/upload", upload.single("file"), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: "No file uploaded" });
  if (!file.mimetype?.includes('pdf')) {
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
    return res.status(400).json({ error: 'File not PDF' });
  }

  const id = uuidv4();
  JOBS[id] = { id, status: "processing", createdAt: Date.now(), questions: [] };

  // Desired counts
  const TARGET = { mcq: 4, saq: 3, laq: 3 };

  try {
    const { text } = await extractTextFromPdf(file.path);
    const cleaned = (text || "").replace(/\s+\n/g, "\n").replace(/\u00A0/g, " ").trim();
    const chunks = chunkText(cleaned, 2500);
    const fallbackChunk = (chunks && chunks.length > 0) ? chunks[0] : cleaned || "No content extracted";

    // --- 1) Batch MCQs (single call) ---
    const mcqChunk = fallbackChunk;
    let mcqItems = [];
    try {
      mcqItems = await generateMultipleMcqs(mcqChunk, TARGET.mcq);
      mcqItems.forEach((parsed, idx) => {
        JOBS[id].questions.push({
          id: `${id}-mcq-${idx}`,
          type: "mcq",
          question: parsed.question || "MCQ question not found",
          options: parsed.options.length === 4 ? parsed.options : parsed.options,
          answer: parsed.correct || parsed.answer || null,
          rawModelOutput: parsed.raw || ""
        });
      });
    } catch (err) {
      console.warn("Batch MCQ error:", err?.message || err);
      // push placeholders for failed MCQs
      for (let i = 0; i < TARGET.mcq; i++) {
        JOBS[id].questions.push({
          id: `${id}-mcq-${i}`,
          type: "mcq",
          error: err?.message || "MCQ generation failed"
        });
      }
    }

    // --- 2) Parallel SAQs ---
    const saqTasks = new Array(TARGET.saq).fill(0).map((_, idx) => async () => {
      // use rotating chunks to get variety; fallback to first chunk
      const chunk = chunks && chunks.length > 0 ? chunks[idx % chunks.length] : fallbackChunk;
      const out = await callModelByType("saq", chunk);
      return {
        id: `${id}-saq-${idx}`,
        type: "saq",
        question: out.question,
        answer: out.answer,
        rawModelOutput: out.raw
      };
    });

    const saqResults = await runWithConcurrency(saqTasks, 3); // concurrency 3
    saqResults.forEach(r => JOBS[id].questions.push(r));

    // --- 3) Parallel LAQs ---
    const laqTasks = new Array(TARGET.laq).fill(0).map((_, idx) => async () => {
      const chunk = chunks && chunks.length > 0 ? chunks[(idx + 1) % chunks.length] : fallbackChunk;
      const out = await callModelByType("laq", chunk);
      const laqLines = (out.raw || "").split(/\r?\n/).map(l => l.trim()).filter(Boolean);
      const laqQuestion = laqLines[0] || "LAQ Question not found.";
      const laqPoints = laqLines.slice(1).filter(l => l.startsWith("-")).map(l => l.substring(1).trim());
      return {
        id: `${id}-laq-${idx}`,
        type: "laq",
        question: laqQuestion,
        answer: laqPoints,
        rawModelOutput: out.raw
      };
    });

    const laqResults = await runWithConcurrency(laqTasks, 2); // concurrency 2 for LAQs
    laqResults.forEach(r => JOBS[id].questions.push(r));

    // cleanup
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
    JOBS[id].status = "completed";

    return res.json({
      jobId: id,
      status: "completed",
      questions: JOBS[id].questions
    });
  } catch (err) {
    console.error("Processing error:", err);
    if (file && fs.existsSync(file.path)) fs.unlinkSync(file.path);
    JOBS[id].status = "failed";
    JOBS[id].error = err.message || String(err);
    return res.status(500).json({ error: err.message || "Processing failed" });
  }
});

/* GET /api/result/:jobId */
app.get("/api/result/:jobId", (req, res) => {
  const job = JOBS[req.params.jobId];
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json(job);
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
