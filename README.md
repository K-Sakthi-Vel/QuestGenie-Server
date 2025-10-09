# QuestGenie Backend Server

This is the Node.js and Express backend for the QuestGenie application. It handles PDF processing, AI-powered question generation using the Google Gemini API, and real-time chat functionality.

## Features Implemented

- **PDF Upload and Processing:** Handles `multipart/form-data` uploads using `multer`, extracts text with `pdf-parse`, and chunks it for the AI model.
- **AI-Powered Quiz Generation:**
  - Generates MCQs, SAQs, and LAQs by sending prompts with text chunks to the Google Gemini API.
  - Uses parallel processing with limited concurrency to generate multiple questions efficiently.
- **YouTube Video Suggestions:** Analyzes PDF content to generate relevant topics and search queries for YouTube, with an optional feature to validate video availability using the YouTube Data API.
- **Real-time Chat Backend:** Provides SSE endpoints to stream AI model responses to the client, enabling a live, interactive chat experience.

## What's Missing

- **Database Integration:** The server is stateless. It does not connect to a database, so it cannot store user data, quiz attempts, scores, or progress.
- **User Authentication:** No authentication or user management system is implemented.
- **Quiz Scoring and Explanations:** The server generates questions but does not include logic for scoring user answers or providing detailed explanations for the correct answers. This would be a necessary addition for a complete learning loop.
- **Persistent State:** All generated content is stored in-memory (`JOBS` object) and will be lost upon server restart.

## Setup

### Prerequisites

- Node.js (v18 or higher is recommended)
- npm (or yarn)

### Installation

1.  Navigate to the `server` directory:
    ```bash
    cd server
    ```
2.  Install the required dependencies:
    ```bash
    npm install
    ```
3.  Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
4.  Open the new `.env` file and add your Google Gemini API key. You can also optionally add a YouTube Data API key to find and validate video suggestions.
    ```
    # Google Gemini API Key (Required)
    GEMINI_API_KEY=your_gemini_api_key_here

    # YouTube Data API v3 Key (Optional)
    GEMINI_YOUTUBE_API_KEY=your_youtube_api_key_here

    # Server port (optional, defaults to 5000)
    PORT=5000
    ```

## How to Run

To start the backend server in development mode with automatic reloading, run the following command from the `server` directory:

```bash
npm run dev
```

The server will start on `http://localhost:5000` (or the port specified in your `.env` file).

For production, you can use:
```bash
npm start
```

## API Endpoints

- `POST /api/upload`: Upload a PDF file to start the question generation process. Returns a `jobId`.
- `GET /api/result/:jobId`: Poll this endpoint to get the status and results of a generation job.
- `POST /api/youtube-suggestions`: Upload a PDF file to get a list of relevant YouTube video suggestions.
- `POST /api/chat/send`: Send a message to the chat endpoint.
- `GET /api/chat/stream/:chatId`: Connect to the Server-Sent Events (SSE) stream to receive real-time chat responses.

## Use of LLM Tools

This server's primary purpose is to orchestrate calls to a Large Language Model (LLM) to power the QuestGenie application.

- **Model Used:** Google Gemini (`gemini-1.5-flash`).
- **Purposes:**
  1.  **Question Generation:** The model is prompted with text extracted from user-uploaded PDFs to generate Multiple Choice, Short Answer, and Long Answer questions.
  2.  **Content Summarization & Topic Extraction:** The model analyzes PDF content to suggest relevant topics for YouTube video searches.
  3.  **Conversational AI:** The model serves as the brain for the real-time chat feature, providing streaming answers to user queries.
