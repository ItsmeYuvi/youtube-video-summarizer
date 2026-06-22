# --- Imports ---
import yt_dlp
import whisper
import os
from google import genai
import gradio as gr

# --- Configuration ---
MAX_DURATION_SECONDS = 420
# Securely grabs the key from your Hugging Face Secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# --- Shared yt-dlp options factory ---
def get_ydl_opts(extra=None):
    """
    Returns a base yt-dlp options dict tuned for cloud server environments
    (Hugging Face Spaces). Uses mobile/Android clients to avoid YouTube's
    bot-detection and SSL connection drops that affect cloud IPs.
    """
    opts = {
        # Use Android client first, fall back to iOS then mweb.
        # These bypass YouTube bot-detection far better than tvhtml5embedded
        # on cloud servers, and avoid SSL UNEXPECTED_EOF errors.
        'extractor_args': {
            'youtube': {
                'client': ['android', 'ios', 'mweb'],
            }
        },
        # Mimic Android YouTube app user-agent for the android client
        'http_headers': {
            'User-Agent': (
                'com.google.android.youtube/19.09.37 '
                '(Linux; U; Android 11) gzip'
            ),
        },
        # SSL / network resilience
        'nocheckcertificate': True,   # handles SSL EOF / cert errors on cloud
        'socket_timeout': 30,
        'retries': 5,
        'fragment_retries': 5,
        'quiet': True,
        'no_warnings': True,
    }
    if extra:
        opts.update(extra)
    return opts


def _parse_ydl_error(e):
    """
    Safely stringify a yt-dlp exception, which can sometimes produce an
    empty str(e) when the DownloadError wraps another exception.
    """
    msg = str(e).strip()
    # Strip the redundant "ERROR: " prefix yt-dlp adds
    if msg.startswith("ERROR: "):
        msg = msg[7:]
    return msg or "Unknown extraction error. Please check the URL and try again."


# --- Core Processing Functions ---
def check_video_duration(youtube_url):
    """Fetches video metadata to check duration without downloading the media."""
    ydl_opts = get_ydl_opts({'skip_download': True})
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_duration = info_dict.get('duration', 0)

            if video_duration > MAX_DURATION_SECONDS:
                return False, (
                    f"Video is too long ({video_duration} seconds). "
                    "Please use a video under 7 minutes to protect server memory."
                )

            return True, video_duration
    except yt_dlp.utils.DownloadError as e:
        error_msg = _parse_ydl_error(e)
        if "Sign in" in error_msg or "confirm your age" in error_msg:
            return False, (
                "⚠️ YouTube requires sign-in for this video. "
                "Please try a different, publicly accessible video."
            )
        if "SSL" in error_msg or "EOF" in error_msg:
            return False, (
                "⚠️ YouTube blocked the cloud server's SSL connection. "
                "Please wait a moment and try again, or try a different video."
            )
        return False, f"Invalid URL or extraction error: {error_msg}"
    except Exception as e:
        return False, f"Unexpected error: {_parse_ydl_error(e)}"

def download_audio(youtube_url):
    """Downloads the best audio stream and converts it to an MP3 file."""
    output_filename = "temp_audio"
    extra = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_filename}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    ydl_opts = get_ydl_opts(extra)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            return True, f"{output_filename}.mp3"
    except yt_dlp.utils.DownloadError as e:
        error_msg = _parse_ydl_error(e)
        if "Sign in" in error_msg or "confirm your age" in error_msg:
            return False, (
                "⚠️ YouTube requires sign-in for this video. "
                "Please try a different, publicly accessible video."
            )
        if "SSL" in error_msg or "EOF" in error_msg:
            return False, (
                "⚠️ YouTube blocked the cloud server's SSL connection during download. "
                "Please wait a moment and try again."
            )
        return False, f"Download failed: {error_msg}"
    except Exception as e:
        return False, f"Unexpected download error: {_parse_ydl_error(e)}"

def transcribe_audio(audio_file_path):
    """Loads the local Whisper model and transcribes the audio file."""
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_file_path)
        transcript_text = result.get("text", "").strip()
        return True, transcript_text
    except Exception as e:
        return False, f"Transcription failed: {str(e)}"

def generate_summary(transcript_text):
    """Sends the raw transcript to Gemini to generate structured bullet points."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        prompt = (
            "You are an expert content summarizer. Below is a raw text transcript extracted "
            "from a YouTube video. Please analyze it thoroughly and generate a clean, "
            "comprehensive summary of the core concepts using bullet points. Use bold headers "
            "where necessary to organize different topics logically.\n\n"
            f"Transcript text:\n{transcript_text}"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return True, response.text
    except Exception as e:
        return False, f"Summarization failed: {str(e)}"

# --- Main Gradio Orchestrator Pipeline ---
def main_ui_pipeline(youtube_url):
    """
    Ties all backend functions together and updates the web UI with statuses/errors.
    """
    if not youtube_url.strip():
        raise gr.Error("Please enter a valid YouTube URL!")

    # Step 1: Validate duration
    is_valid, duration_result = check_video_duration(youtube_url)
    if not is_valid:
        raise gr.Error(duration_result)
        
    # Step 2: Download Audio stream
    gr.Info("⬇️ Fetching video and downloading audio stream... Please wait.")
    download_success, file_result = download_audio(youtube_url)
    if not download_success:
        raise gr.Error(file_result)
        
    # Step 3: Local Whisper Transcription
    gr.Info("🎙️ Audio downloaded successfully! Running speech-to-text transcription engine...")
    transcribe_success, transcript_result = transcribe_audio(file_result)
    
    # Instant local file deletion cleanup right after transcribing
    if os.path.exists(file_result):
        os.remove(file_result)
        
    if not transcribe_success:
        raise gr.Error(transcript_result)
        
    # Step 4: LLM Analysis & Bulleted Generation
    gr.Info("🧠 Processing transcript text with Gemini AI to extract core concepts...")
    summary_success, summary_result = generate_summary(transcript_result)
    if not summary_success:
        raise gr.Error(summary_result)
        
    # Everything succeeded! Return the markdown response text directly to the UI panel
    return summary_result

# --- Gradio UI Layout Building ---
with gr.Blocks(title="AI YouTube Summarizer") as demo:
    gr.Markdown("# 🎥 Automated YouTube Video Summarizer")
    gr.Markdown("Paste a YouTube link below to download its audio, convert speech to text, and get a structured summary.")
    gr.Markdown("> **Note:** To prevent server memory exhaustion, maximum video length is strictly capped at **7 minutes**.")
    
    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube Video Link", 
            placeholder="https://www.youtube.com/watch?v=...", 
            lines=1
        )
        
    submit_btn = gr.Button("🚀 Generate Summary", variant="primary")
    
    output_markdown = gr.Markdown()

    submit_btn.click(
        fn=main_ui_pipeline, 
        inputs=url_input, 
        outputs=output_markdown
    )

# --- App Launch Engine ---
if __name__ == "__main__":
    demo.launch()