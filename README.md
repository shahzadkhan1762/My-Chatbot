
# My AI Agent — Fine-Tuned TinyLlama Chatbot

A locally running AI chatbot built with **Python, FastAPI, TinyLlama, and LoRA**. The project combines a modern, ChatGPT-inspired web interface with a REST API, allowing users to interact with a fine-tuned language model through a browser or API requests.

## Features

- **AI-powered conversations** using TinyLlama with a LoRA adapter.
- **REST API** built with FastAPI.
- **Interactive API documentation** through Swagger UI.
- **Modern web interface** inspired by ChatGPT.
- **Chat history** saved in the browser.
- **Light and dark modes**.
- **Copy and export** conversation content.
- **Voice input and read-aloud** using supported browser features.
- **Image attachment previews** in the chat interface.
- **Health endpoint** for checking the backend and model status.
- **Local execution** without requiring a hosted AI API.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend programming language |
| FastAPI | API development |
| Uvicorn | ASGI web server |
| TinyLlama | Base language model |
| Hugging Face Transformers | Model loading and text generation |
| PEFT / LoRA | Loading the fine-tuning adapter |
| PyTorch | Model inference |
| HTML, CSS, JavaScript | Frontend interface |

## Project Structure

```text
My-Chatbot/
├── backend.py
├── index.html
├── models/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── venv313/
├── requirements.txt
└── README.md
```

The `venv313/` directory contains the local Python virtual environment and should not be uploaded to GitHub.

## Requirements

- Python 3.13, subject to compatibility with the installed ML packages
- Windows, Linux, or macOS with suitable dependencies
- Sufficient RAM and storage for the language model
- Internet access for downloading the base model if it is not cached locally

A computer with a compatible GPU may improve inference speed. CPU inference is also possible, but can be slow.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/My-Chatbot.git
cd My-Chatbot
```

Replace `YOUR-USERNAME` with your GitHub username and adjust the repository name if necessary.

### 2. Create a virtual environment

```bash
python -m venv venv313
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\venv313\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
venv313\Scripts\activate.bat
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If you encounter dependency compatibility issues, install compatible versions of PyTorch, Transformers, and PEFT for your Python version.

### 5. Configure the model

The adapter in `models/` is designed for the base model:

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Ensure the adapter files are present and the model path in `backend.py` matches the available files.

If the base model is not cached, the application may need an internet connection to download it from Hugging Face.

### 6. Start the API server

From the project root, run:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

Wait for the model to load and for Uvicorn to report that the server is running.

## Access the Application

Once the server is running, open these addresses in your browser:

| Resource | URL |
|---|---|
| Chatbot interface | http://127.0.0.1:8000/ |
| Interactive API documentation | http://127.0.0.1:8000/docs |
| OpenAPI schema | http://127.0.0.1:8000/openapi.json |
| Health endpoint | http://127.0.0.1:8000/health |

Keep the terminal running while using the application.

## API Usage

### Check API health

**Endpoint:** `GET /health`

Use this endpoint to check the backend and model status.

### Send a chat message

**Endpoint:** `POST /chat`

Example request:

```json
{
  "message": "Hello! Please introduce yourself."
}
```

Example response format:

```json
{
  "reply": "Hello! I am your AI assistant."
}
```

The actual response depends on the loaded model and its generation settings.

You can test the endpoint directly from the interactive Swagger UI at `/docs`.

## Important Notes

- The API currently runs locally by default and is not automatically accessible from the public internet.
- Image attachments currently provide frontend previews; image understanding requires a compatible vision model and backend implementation.
- Voice features depend on browser support and permissions.
- Response speed depends on hardware, model configuration, and generated response length.
- Keep large model files and virtual environments out of Git unless you intentionally use an appropriate model-hosting solution.
- The LoRA adapter requires its compatible base model.

## Future Improvements

- API-key authentication and access control
- Cloud deployment
- Improved inference speed and memory efficiency
- Vision-model integration for image understanding
- Persistent server-side conversation memory
- Automated tests and API error handling
- Streaming responses for a more interactive chat experience

## License

Choose a license before distributing this project publicly. If you intend to use an existing model or adapter, also review their respective licenses and usage conditions.

---

**Developed as a personal AI chatbot project using TinyLlama, LoRA, Python, and FastAPI.**
