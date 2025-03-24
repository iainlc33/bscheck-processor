from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
import requests
import openai
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the FastAPI application
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
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "your_apify_token")

# Initialize clients
openai.api_key = OPENAI_API_KEY

class TikTokRequest(BaseModel):
    url: str

TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_tiktok_video(url):
    """Download TikTok video using apidojo/tiktok-scraper"""
    try:
        logger.info(f"Trying to download video from {url}")
        file_id = str(int(time.time()))
        output_path = f"{TEMP_DIR}/{file_id}.mp4"
        
        # Format exactly matching the documentation example
        run_input = {
            "startUrls": [url],
            "maxItems": 1
        }
        
        # Log the exact input being sent for debugging
        logger.info(f"Sending Apify request with input: {run_input}")
        
        # Make API request to Apify
        run_url = "https://api.apify.com/v2/acts/apidojo~tiktok-scraper/runs"
        
        headers = {
            "Authorization": f"Bearer {APIFY_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Start the scraper run
        response = requests.post(run_url, headers=headers, json=run_input)
        
        if response.status_code != 201:
            logger.error(f"Apify run creation failed: {response.text}")
            raise Exception(f"Failed to start Apify scraper: {response.status_code}")
        
        run_data = response.json()
        run_id = run_data["data"]["id"]
        
        # Log the run URL for manual checking
        logger.info(f"Apify run created: https://console.apify.com/view/runs/{run_id}")
        
        # Wait for the run to finish
        run_status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
        
        # Poll for completion (with timeout)
        start_time = time.time()
        timeout = 90  # seconds - increased timeout for TikTok scraping
        
        while time.time() - start_time < timeout:
            status_response = requests.get(run_status_url, headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"})
            status_data = status_response.json()
            
            status = status_data["data"]["status"]
            logger.info(f"Apify run status: {status}")
            
            if status == "SUCCEEDED":
                break
                
            if status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                status_message = status_data["data"].get("statusMessage", "No status message")
                logger.error(f"Apify run failed. Status: {status}, Message: {status_message}")
                logger.error(f"Apify run details: {status_data}")
                raise Exception(f"Apify run failed with status: {status}")
                
            # Wait before checking again
            time.sleep(5)
        
        # Check if timeout occurred
        if time.time() - start_time >= timeout:
            logger.error("Apify run timed out")
            raise Exception("Apify run timed out")
        
        # Get the results from the dataset
        dataset_id = status_data["data"]["defaultDatasetId"]
        dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
        
        results_response = requests.get(dataset_url, headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"})
        results = results_response.json()
        
        if not results or len(results) == 0:
            raise Exception("No results found from Apify scraper")
        
        # Log the result structure to help debugging
        logger.info(f"Apify returned {len(results)} results")
        logger.info(f"First result keys: {list(results[0].keys())}")
        
        # Directly extract video URL from the expected location
        video_item = results[0]
        video_url = None
        
        if 'video' in video_item and 'url' in video_item['video']:
            video_url = video_item['video']['url']
            logger.info(f"Found video URL: {video_url}")
        else:
            logger.error("Could not find video URL in expected location")
            logger.error(f"Full first result: {video_item}")
            raise Exception("No video URL found in Apify results")
        
        # Download the video
        logger.info(f"Downloading video from URL: {video_url}")
        video_response = requests.get(video_url, stream=True)
        
        if video_response.status_code != 200:
            raise Exception(f"Failed to download video: Status code {video_response.status_code}")
            
        with open(output_path, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        # Verify file was downloaded
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Video downloaded successfully to {output_path}")
            return output_path
        else:
            raise Exception("Video file is empty or doesn't exist after download")
        
    except Exception as e:
        logger.error(f"Error extracting TikTok: {str(e)}")
        raise Exception(f"Failed to extract TikTok video: {str(e)}")

def extract_audio(video_path):
    """Convert video to audio file and return the file path"""
    try:
        audio_path = f"{video_path}.mp3"
        command = f"ffmpeg -i '{video_path}' -q:a 0 -map a '{audio_path}' -y"
        
        # Execute FFmpeg command and capture output for potential debugging
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {process.stderr}")
            raise Exception(f"FFmpeg failed with return code {process.returncode}")
        
        # Verify the audio file was created successfully
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return audio_path
        else:
            logger.error("FFmpeg completed but audio file is missing or empty")
            raise Exception("Failed to extract audio: output file is missing or empty")
            
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise Exception(f"Failed to extract audio: {str(e)}")

def transcribe_audio(audio_path):
    """Use OpenAI Whisper to transcribe audio"""
    try:
        # Verify the audio file exists before transcribing
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        with open(audio_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")

def analyze_content(transcript):
    """Use OpenAI to analyze the content"""
    try:
        system_prompt = """You are a professional fact-checker and researcher, dedicated to systematically analyzing claims with scientific rigor and precision. 

Your task is to:
1. Identify specific claims in the transcript
2. Methodically evaluate each claim using credible, verifiable sources
3. Provide a structured, detailed analysis that:
   - Directly quotes the original claim
   - Rates the claim's accuracy
   - Provides clear, evidence-based reasoning
   - Cites reputable sources for each refutation or verification

Formatting Guidelines:
- Use a numbered list for each claim
- For each claim, include:
  a) Exact claim quotation
  b) Accuracy rating (🚨 PURE BS, 🐂 HEAVY BS, 🤥 MODERATE BS, 🤨 SLIGHT BS, 💯 NO BS)
  c) Detailed explanation of why the claim is true/false
  d) Credible sources that support or refute the claim

Source Credibility Hierarchy:
- Peer-reviewed scientific journals
- Government health/scientific agencies
- Reputable academic institutions
- Respected medical/scientific organizations

Tone: Professional, direct, and unequivocal. Focus on facts, not humor.

Example Format:
1. Claim: "[Verbatim claim from transcript]"
   Rating: [Emoji + Rating]
   Explanation: Detailed analysis of claim's accuracy
   Sources: 
   - [Credible Source 1]
   - [Credible Source 2]

Your ultimate goal is to provide a clear, authoritative breakdown that exposes misinformation and reinforces factual understanding."""

        # Use OpenAI as the primary model
        response = openai.chat.completions.create(
            model="gpt-4-turbo", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TikTok transcript: {transcript}"}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        logger.error(f"Error analyzing content: {str(e)}")
        raise Exception(f"Failed to analyze content: {str(e)}")

@app.post("/process")
async def process_tiktok_endpoint(request: TikTokRequest):
    """Endpoint to process a TikTok URL"""
    # Special case for test mode - check before URL validation
    if request.url == "test" or request.url == "https://example.com" or "test" in request.url.lower():
        logger.info("Using test mode with mock data")
        transcript = "This is a test transcript from a TikTok video. In this video, someone is claiming that drinking lemon water every morning will make you lose 10 pounds in a week without exercise. They also say that celebrities use this secret trick all the time but the diet industry doesn't want you to know about it."
        response_text = analyze_content(transcript)
        return {
            "status": "success", 
            "transcript": transcript, 
            "response": response_text
        }
    
    # Validate the request for non-test URLs
    if not request.url or not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    video_path = None
    audio_path = None
    
    try:
        # Log the beginning of processing
        logger.info(f"Processing URL: {request.url}")
        
        # Extract the video
        logger.info("Extracting video using Apify...")
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
        logger.info("Analyzing content...")
        response_text = analyze_content(transcript)
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
