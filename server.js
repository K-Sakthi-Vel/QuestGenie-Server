/** 
 * server.js
 * Node/Express backend for PDF -> Text -> Claude QG (MCQ, SAQ, LAQ)
 * Usage:
 * npm install
 * cp .env.example .env   (fill HF_API_TOKEN)
 * npm run start
 *
 * Endpoints:
 * POST /api/upload  (multipart form, field "file")
 * GET  /api/result/:jobId
 */

const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const pdf = require("pdf-parse");
const fetch = require("node-fetch");
const cors = require("cors");
const dotenv = require("dotenv");
const { v4: uuidv4 } = require("uuid");
dotenv.config();

const HF_API_TOKEN = process.env.HF_API_TOKEN;
if (!HF_API_TOKEN) {
  console.error("Missing HF_API_TOKEN in environment. See .env.example");
  process.exit(1);
}
console.log("Using HF_API_TOKEN:", HF_API_TOKEN);

const PORT = process.env.PORT || 5000;
const app = express();
app.use(express.json());
app.use(cors());

// Multer setup (store to /tmp)
const upload = multer({
  dest: path.join(process.cwd(), "tmp"),
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB max
});

// In-memory job store (use DB in prod)
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

/* Chunk text by maxChars */
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
  return `Generate exactly one Multiple Choice Question (MCQ) with 4 options (A, B, C, D) and specify the correct answer from the following passage:

Passage: ${chunk}

Format:
Question: <question_text>
A) <option A>
B) <option B>
C) <option C>
D) <option D>
Answer: <A|B|C|D>`;
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

/* Parse MCQ output */
function parseMcqOutput(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const result = { question: null, options: [], answer: null, raw: text };
  for (const line of lines) {
    if (!result.question && line.toLowerCase().startsWith("question:")) {
      result.question = line.replace(/^question:\s*/i, "");
      continue;
    }
    const matchOpt = line.match(/^[A-D]\)\s*(.+)$/i);
    if (matchOpt) {
      result.options.push(matchOpt[1].trim());
      continue;
    }
    const ansMatch = line.match(/^answer:\s*([A-D])/i);
    if (ansMatch) result.answer = ansMatch[1].toUpperCase();
  }
  if (result.answer && result.options.length === 4) {
    const idx = result.answer.charCodeAt(0) - 65;
    result.correct = result.options[idx];
  }
  return result;
}

/* Call OpenRouter Claude model */
async function callClaudeModel(prompt, maxTokens = 500) {
  const url = "https://openrouter.ai/api/v1/chat/completions";
  const headers = {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${HF_API_TOKEN}`
  };
  const body = {
    model: "anthropic/claude-sonnet-4.5",
    messages: [
      { role: "system", content: "You are a helpful assistant that generates quiz questions from text." },
      { role: "user", content: prompt }
    ],
    max_tokens: maxTokens
  };

  const resp = await fetch(url, { method: "POST", headers, body: JSON.stringify(body), timeout: 120000 });
  if (!resp.ok) {
    const text = await resp.text();
    const err = new Error(`Claude API error ${resp.status}: ${text}`);
    if (resp.status === 429) err.isRateLimit = true;
    throw err;
  }

  const json = await resp.json();
  return json?.choices?.[0]?.message?.content?.trim() || "";
}

/* Unified call by type */
async function callModelByType(type, chunk) {
  if (type === "saq") {
    const prompt = `Generate a short-answer question and answer from the following text:\n\n${chunk}`;
    const raw = await callClaudeModel(prompt, 300);
    const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const question = lines[0] || "SAQ Question not found";
    const answer = lines.slice(1).join(" ") || "Answer not found";
    return { question, answer, raw };
  } else if (type === "mcq") {
    const raw = await callClaudeModel(mcqPromptFromChunk(chunk), 500);
    return { raw };
  } else if (type === "laq") {
    const raw = await callClaudeModel(laqPromptFromChunk(chunk), 700);
    return { raw };
  }
  throw new Error(`Unknown type: ${type}`);
}

/* Root endpoint */
app.get("/", (req, res) => res.send("PDF QG backend"));

/* POST /api/upload */
app.post("/api/upload", upload.single("file"), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: "No file uploaded" });
  if (!file.mimetype?.includes('pdf')) { fs.unlinkSync(file.path); return res.status(400).json({ error: 'File not PDF' }); }

  const id = uuidv4();
  JOBS[id] = { id, status: "processing", createdAt: Date.now(), questions: [] };

  try {
    const { text, pages = [], metadata } = await extractTextFromPdf(file.path);
    const cleaned = text.replace(/\s+\n/g, "\n").replace(/\u00A0/g, " ").trim();
    const chunks = chunkText(cleaned, 2500);
    let qCounter = 0;

    for (const chunk of chunks) {
      if (qCounter >= 30) break;

      // MCQ
      try {
        const mcqOut = await callModelByType("mcq", chunk);
        const parsed = parseMcqOutput(mcqOut.raw);
        JOBS[id].questions.push({ id: `${id}-mcq-${qCounter}`, type: "mcq", question: parsed.question, options: parsed.options, answer: parsed.correct || parsed.answer, rawModelOutput: mcqOut.raw });
        qCounter++;
      } catch (err) { console.warn("MCQ error:", err.message); }

      // SAQ
      try {
        const saqOut = await callModelByType("saq", chunk);
        JOBS[id].questions.push({ id: `${id}-saq-${qCounter}`, type: "saq", question: saqOut.question, answer: saqOut.answer, rawModelOutput: saqOut.raw });
        qCounter++;
      } catch (err) { console.warn("SAQ error:", err.message); }

      // LAQ
      try {
        const laqOut = await callModelByType("laq", chunk);
        const laqLines = laqOut.raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
        let question = "LAQ Question not found.";
        const expectedPoints = [];
        let inPointsSection = false;

        for (const line of laqLines) {
          if (line.toLowerCase().startsWith("question:")) {
            question = line.replace(/^question:\s*/i, "").trim();
          } else if (line.toLowerCase().startsWith("expected points:")) {
            inPointsSection = true;
          } else if (inPointsSection && line.startsWith("-")) {
            expectedPoints.push(line.substring(1).trim());
          }
        }

        JOBS[id].questions.push({ id: `${id}-laq-${qCounter}`, type: "laq", question, expected_points: expectedPoints, rawModelOutput: laqOut.raw });
        qCounter++;
      } catch (err) { console.warn("LAQ error:", err.message); }

      await new Promise(r => setTimeout(r, 300));
    }

    JOBS[id].status = "done";
    JOBS[id].metadata = metadata;
    fs.unlinkSync(file.path);
    return res.json({ jobId: id, questions: JOBS[id].questions });

  } catch (err) {
    JOBS[id].status = "error";
    JOBS[id].error = String(err);
    console.error("Processing error:", err);
    return res.status(500).json({ error: String(err) });
  }
});

/* GET /api/result/:jobId */
app.get("/api/result/:jobId", (req, res) => {
  const job = JOBS[req.params.jobId];
  if (!job) return res.status(404).json({ error: "job not found" });
  return res.json(job);
});

app.listen(PORT, () => console.log(`PDF QG backend running at http://localhost:${PORT}`));
