from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import yt_dlp
import os
import subprocess
import requests
import openai
from anthropic import Anthropic
import boto3
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_key")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your_anthropic_key")
MAKE_CALLBACK_URL = os.environ.get("MAKE_CALLBACK_URL", "your_make_callback_url")

# Initialize clients
openai.api_key = OPENAI_API_KEY
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

class TikTokRequest(BaseModel):
    url: str
    mode: str
    webhook_id: str  # Add this to identify which Make.com webhook execution to callback to

class ProcessingResult(BaseModel):
    transcript: str
    response: str
    url: str
    mode: str
    webhook_id: str

TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_tiktok_video(url):
    """Download TikTok video and return the file path"""
    try:
        # Generate a unique filename
        file_id = str(uuid.uuid4())
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{TEMP_DIR}/{file_id}.%(ext)s',
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = f"{TEMP_DIR}/{file_id}.{info['ext']}"
            return file_path
    except Exception as e:
        logger.error(f"Error extracting TikTok: {str(e)}")
        raise Exception(f"Failed to extract TikTok video: {str(e)}")

def extract_audio(video_path):
    """Convert video to audio file and return the file path"""
    try:
        audio_path = f"{video_path}.mp3"
        command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
        subprocess.call(command, shell=True)
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise Exception(f"Failed to extract audio: {str(e)}")

def transcribe_audio(audio_path):
    """Use OpenAI Whisper to transcribe audio"""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

def analyze_content(transcript, mode):
    """Use Claude to analyze the content"""
    try:
        if mode.lower() == "fact_check":
            system_prompt = """
            You are a skeptical fact-checker analyzing TikTok videos. 
            Examine the transcript for factual claims.
            Research each claim and provide a clear verdict. 
            Format your response in easy-to-read sections with verdicts clearly marked.
            Keep your response friendly but direct.
            """
        else:  # roast mode
            system_prompt = """
            You are a witty comedian specializing in roasts. 
            Your job is to create a funny, snarky response to this TikTok content.
            Be clever, not mean-spirited. Focus on the content, not personal attacks.
            Keep it to 3-4 sentences maximum - short and biting.
            Use casual, internet-savvy language that would resonate with TikTok users.
            """
        
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=750,
            system=system_prompt,
            messages=[{"role": "user", "content": f"TikTok transcript: {transcript}"}]
        )
        
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error analyzing content: {str(e)}")
        raise Exception(f"Failed to analyze content: {str(e)}")

def cleanup_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")

async def process_tiktok(request: TikTokRequest):
    """Process a TikTok URL and return analysis"""
    video_path = None
    audio_path = None
    
    try:
        # Extract the video
        video_path = extract_tiktok_video(request.url)
        
        # Extract audio
        audio_path = extract_audio(video_path)
        
        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        
        # Analyze content
        response = analyze_content(transcript, request.mode)
        
        # Send results back to Make.com
        result = ProcessingResult(
            transcript=transcript,
            response=response,
            url=request.url,
            mode=request.mode,
            webhook_id=request.webhook_id
        )
        
        # Callback to Make.com with the results
        callback_response = requests.post(
            MAKE_CALLBACK_URL,
            json=result.dict()
        )
        
        if callback_response.status_code != 200:
            logger.error(f"Failed to send callback: {callback_response.text}")
            
        return {"status": "success", "message": "Processing completed"}
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        # Attempt to notify Make.com of the failure
        try:
            error_result = {
                "error": str(e),
                "url": request.url,
                "mode": request.mode,
                "webhook_id": request.webhook_id
            }
            requests.post(MAKE_CALLBACK_URL, json=error_result)
        except:
            pass
        return {"status": "error", "message": str(e)}
    
    finally:
        # Clean up temporary files
        files_to_cleanup = [f for f in [video_path, audio_path] if f]
        cleanup_files(files_to_cleanup)

@app.post("/process")
async def process_tiktok_endpoint(request: TikTokRequest, background_tasks: BackgroundTasks):
    """Endpoint to process a TikTok URL"""
    # Validate the request
    if not request.url or not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    if not request.mode or request.mode.lower() not in ["fact_check", "roast"]:
        # Default to roast if not specified correctly
        request.mode = "roast"
    
    # Process in background to avoid timeouts
    background_tasks.add_task(process_tiktok, request)
    
    return {"status": "processing", "message": "Your request is being processed"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
