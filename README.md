# Audio Transcription & Q&A System

A powerful AI-powered web application that can download YouTube videos, transcribe audio content, and allow you to ask questions about the transcribed text using advanced AI models.

## 🌟 What This System Does

This application combines several AI technologies to help you:
- **Download audio** from YouTube videos
- **Transcribe speech** to text with high accuracy
- **Ask questions** about the transcribed content
- **Generate summaries** of long audio content

## 🚀 Features

### 1. Audio Transcription
- Download audio from any YouTube URL
- Transcribe audio in multiple languages
- Translate foreign language audio to English
- High-accuracy speech recognition

### 2. Intelligent Q&A
- Ask questions about your transcribed content
- Get context-aware answers powered by AI
- See relevant source text for each answer
- Semantic search through your content

### 3. Content Summarization
- Generate concise summaries of long audio
- Extract key points and important information
- Multiple summary formats available

## 📋 Prerequisites

Before you start, make sure you have:

1. **Python 3.8 or higher** installed on your computer
2. **FFmpeg** installed for audio processing
3. **At least 8GB RAM** (16GB recommended for better performance)
4. **Google API Key** for the AI models

## 🛠️ Installation Steps

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd major_project

# Or simply download and extract the project folder
```

### Step 2: Install Python Dependencies
Open your terminal/command prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3: Install FFmpeg (Required for Audio Processing)

**For Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract the files to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your system PATH

**For macOS:**
```bash
brew install ffmpeg
```

**For Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### Step 4: Get Google API Key
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy the API key (you'll need it later)

## 🚀 How to Run the Application

### Step 1: Start the Application
Open your terminal in the project folder and run:
```bash
python app.py
```

You should see output like:
```
Loading Whisper model...
✅ Whisper model loaded successfully
Loading embedding model...
✅ Embedding model loaded successfully
Loading text splitter...
✅ Text splitter loaded successfully
Loading LLM model...
✅ LLM model loaded successfully
🎉 All models loaded successfully!
 * Running on http://0.0.0.0:5000
```

### Step 2: Open Your Web Browser
Go to: `http://localhost:5000`

You should see the home page of the application.

## 📖 How to Use the Application

### Step 1: Transcribe Audio
1. Click on **"Transcribe"** in the navigation menu
2. Paste a YouTube URL in the input field
3. Choose whether to transcribe in original language or translate to English
4. Click **"Download & Process"**
5. Wait for the audio to download and transcribe
6. The transcribed text will appear on the page

### Step 2: Process for Q&A
1. After transcription, click **"Process for Q&A"**
2. This creates searchable indexes of your content
3. Wait for processing to complete

### Step 3: Ask Questions
1. Click on **"Q&A"** in the navigation menu
2. Type your question about the transcribed content
3. Click **"Ask Question"**
4. Get AI-powered answers with relevant source text

### Step 4: Generate Summaries
1. Click on **"Summarize"** in the navigation menu
2. Enter the name of your transcribed text file
3. Choose summary type (general, detailed, etc.)
4. Click **"Generate Summary"**

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Models not loaded" Error
**Problem:** Application shows models failed to load
**Solution:** 
- Close other applications to free up memory
- Increase your system's virtual memory
- Restart the application

#### 2. "FFmpeg not found" Error
**Problem:** Audio download fails
**Solution:**
- Make sure FFmpeg is installed correctly
- Check that FFmpeg is in your system PATH
- Try downloading without audio conversion

#### 3. "Unexpected token '<'" Error
**Problem:** JavaScript error when clicking buttons
**Solution:**
- This has been fixed in the latest version
- Make sure you're using the updated code
- Check that the server is running properly

#### 4. Memory Issues
**Problem:** Application crashes or models fail to load
**Solution:**
- Close other applications (browsers, games, etc.)
- Increase system virtual memory
- Use a computer with more RAM (16GB+ recommended)

### Performance Tips
- **Close unnecessary applications** before running
- **Use a computer with 16GB+ RAM** for best performance
- **Wait for models to load completely** before using features
- **Process one video at a time** to avoid memory issues

## 📁 Project Structure

```
major_project/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/               # AI model configurations
│   ├── whisper_model.py
│   ├── embedding_model.py
│   ├── llm_model.py
│   └── text_splitter.py
├── services/             # Business logic
│   ├── transcription_service.py
│   ├── embedding_service.py
│   ├── qa_service.py
│   └── summarize_service.py
├── templates/            # HTML templates
│   ├── base.html
│   ├── home.html
│   ├── transcribe.html
│   ├── qa.html
│   └── summarize.html
└── utils/               # Utility functions
    └── file_utils.py
```

## 🔒 Privacy and Security

- **Local Processing:** All AI processing happens on your computer
- **No Data Upload:** Your audio and text stay on your machine
- **API Keys:** Store your Google API key securely
- **Temporary Files:** Audio and text files are saved locally

## 🆘 Getting Help

If you encounter issues:

1. **Check the console output** for error messages
2. **Verify all prerequisites** are installed correctly
3. **Ensure sufficient memory** is available
4. **Check your internet connection** for YouTube downloads
5. **Verify your Google API key** is valid

## 🎯 Example Workflow

1. **Find a YouTube video** you want to analyze
2. **Copy the URL** (e.g., https://www.youtube.com/watch?v=example)
3. **Open the application** in your browser
4. **Go to Transcribe page** and paste the URL
5. **Click Download & Process** and wait
6. **Click Process for Q&A** to make it searchable
7. **Go to Q&A page** and ask questions about the content
8. **Get intelligent answers** with source references

## 🔄 Updates and Maintenance

- **Regular Updates:** Keep Python packages updated
- **Model Updates:** AI models may be updated periodically
- **Backup:** Keep copies of important transcribed files
- **Cleanup:** Remove old audio files to save space

## 📝 License

This project is for educational and personal use. Please respect YouTube's terms of service and copyright laws when downloading content.

---

**Happy Transcribing! 🎤✨** 
