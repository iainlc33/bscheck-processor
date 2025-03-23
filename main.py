from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
import os
import subprocess
import openai
from anthropic import Anthropic
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow requests from your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_key")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your_anthropic_key")

# Initialize clients
openai.api_key = OPENAI_API_KEY
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

class TikTokRequest(BaseModel):
    url: str
    mode: str = "roast"  # Default to roast if not specified

TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_tiktok_video(url):
    """Download TikTok video and return the file path"""
    try:
        # Generate a unique filename using timestamp
        file_id = str(int(time.time()))
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{TEMP_DIR}/{file_id}.%(ext)s',
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Get the downloaded file path
            for entry in os.scandir(TEMP_DIR):
                if entry.name.startswith(file_id):
                    return entry.path
            
            # Fallback in case we can't find the exact file
            return f"{TEMP_DIR}/{file_id}.{info.get('ext', 'mp4')}"
    except Exception as e:
        logger.error(f"Error extracting TikTok: {str(e)}")
        raise Exception(f"Failed to extract TikTok video: {str(e)}")

def extract_audio(video_path):
    """Convert video to audio file and return the file path"""
    try:
        audio_path = f"{video_path}.mp3"
        command = f"ffmpeg -i '{video_path}' -q:a 0 -map a '{audio_path}' -y"
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

@app.post("/process")
async def process_tiktok_endpoint(request: TikTokRequest):
    """Endpoint to process a TikTok URL"""
    # Validate the request
    if not request.url or not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    video_path = None
    audio_path = None
    
    try:
        # Log the beginning of processing
        logger.info(f"Processing URL: {request.url}, Mode: {request.mode}")
        
        # Extract the video
        logger.info("Extracting video...")
        video_path = extract_tiktok_video(request.url)
        logger.info(f"Video extracted to: {video_path}")
        
        # Extract audio
        logger.info("Extracting audio...")
        audio_path = extract_audio(video_path)
        logger.info(f"Audio extracted to: {audio_path}")
        
        # Transcribe audio
        logger.info("Transcribing audio...")
        transcript = transcribe_audio(audio_path)
        logger.info(f"Transcription: {transcript}")
        
        # Analyze content
        logger.info(f"Analyzing content with mode: {request.mode}")
        response_text = analyze_content(transcript, request.mode)
        logger.info("Analysis complete")
        
        # Return the response
        return {
            "status": "success", 
            "transcript": transcript, 
            "response": response_text
        }
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up files
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "BSCheck TikTok Processing API", "status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
