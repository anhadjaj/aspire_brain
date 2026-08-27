import os
import json
import base64
from datetime import datetime, timezone

from flask import Flask, request, Response, jsonify

from groq import Groq
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from newsapi import NewsApiClient

# --- SECURITY & API CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GLASSES_SECRET = os.getenv("GLASSES_SECRET")

if not all([GROQ_API_KEY, NEWS_API_KEY, GLASSES_SECRET]):
    raise RuntimeError("CRITICAL: Missing environment variables! Check Render settings.")

client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)

SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/drive.readonly'
]

# --- GOOGLE & NEWS INTEGRATIONS ---
class GoogleServices:
    def __init__(self):
        self.creds = None
        # Load from Render's Secret Files
        if os.path.exists('token.json'):
            self.creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                self.creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(self.creds.to_json())
                
        self.gmail = build('gmail', 'v1', credentials=self.creds)
        self.calendar = build('calendar', 'v3', credentials=self.creds)
        print("[System]: Google APIs Authenticated.")

google_services = GoogleServices()
news_client = NewsApiClient(api_key=NEWS_API_KEY)

# --- GLOBAL CHAT HISTORY ---
current_context = datetime.now().astimezone().strftime("%A, %B %d, %Y %I:%M %p %z")
SYSTEM_PROMPT = (
    "You are an articulate, engaging, and conversational AI assistant running on a pair of smart glasses. Your name is VIPER. "
  "Your responses are spoken aloud directly into the ear of your user, Anhad. "
  "Speak naturally and expressively, like a knowledgeable friend. "
  "Stay focused on the user's intent and provide only information relevant to fulfilling it. "
  "Keep responses concise by default, and provide additional detail only when it is useful or requested. "
  "CAMERA ORIENTATION RULE: The images provided to you are rotated 90 degrees counter-clockwise because of how the camera is mounted. You must mentally rotate the image 90 degrees clockwise before analyzing it. What appears on the left side of the image is actually the bottom/floor, and what appears on the right side is the top/ceiling. Read any text as if it were rotated correctly. "
  "LIVE TRANSLATION RULE: Act as a real-time visual translator. When asked to read or translate text, output ONLY the direct English translation of the visible text. Provide absolutely no introductions, context, or conversational filler. Just the translated words. "
  "VISION RULE: When asked about what you see, hyper-focus ONLY on the primary subject relevant to the user's request. "
  "Do not describe the background, room, user's body, lighting, or unrelated objects unless explicitly asked. "
  "UNCERTAINTY RULE: Never invent, assume, or hallucinate visual details. "
  "If the subject is unclear, obscured, distant, or unreadable, say so rather than guessing. "
  "Do not use filler phrases like 'I see'. Do not use markdown, emojis, asterisks, or special formatting. "
  "For time based outputs don't say 7 o clock pm, instead say 7 p.m. or 7 in the evening. "
  "For dates, say 'August 16th' instead of 'August 16'."
)
chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# --- TOOLS SCHEMA ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_unread_emails",
            "description": "Fetch the user's latest unread emails from Gmail.",
            "parameters": {"type": "object", "properties": {"max_results": {"type": "integer"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_events",
            "description": "Fetch upcoming schedule from Google Calendar.",
            "parameters": {"type": "object", "properties": {"max_results": {"type": "integer"}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetch top news headlines for a specific category and country.",
            "parameters": {"type": "object", "properties": {"category": {"type": "string"}, "country": {"type": "string"}}}
        }
    }
]

def execute_tool(name, args):
    try:
        if name == "get_unread_emails":
            res = google_services.gmail.users().messages().list(userId='me', labelIds=['INBOX', 'UNREAD'], maxResults=args.get("max_results", 3)).execute()
            messages = res.get('messages', [])
            if not messages: return "You have no unread emails."
            
            email_summaries = []
            for msg in messages:
                txt = google_services.gmail.users().messages().get(userId='me', id=msg['id'], format='metadata', metadataHeaders=['From', 'Subject']).execute()
                headers = txt['payload']['headers']
                sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
                email_summaries.append(f"From: {sender}, Subject: {subject}")
            return "\n".join(email_summaries)
            
        elif name == "get_upcoming_events":
            now = datetime.now(timezone.utc).isoformat()
            events_result = google_services.calendar.events().list(calendarId='primary', timeMin=now, maxResults=args.get("max_results", 5), singleEvents=True, orderBy='startTime').execute()
            events = events_result.get('items', [])
            if not events: return "You have no upcoming events on your calendar."
            
            event_list = [f"'{e.get('summary')}' at {e['start'].get('dateTime', e['start'].get('date'))}" for e in events]
            return "Upcoming events:\n" + "\n".join(event_list)
            
        elif name == "get_news":
            top_headlines = news_client.get_top_headlines(category=args.get("category", "general"), language='en')
            articles = top_headlines.get('articles', [])
            if not articles: return "No headlines found."
            return "Headlines: " + ". ".join([a.get('title', 'Untitled') for a in articles[:3]])
            
        return "Tool executed."
    except Exception as e:
        return f"Error executing tool {name}: {str(e)}"

# --- GROQ TEXT-TO-SPEECH ---
def generate_tts_audio(text: str) -> bytes:
    """Uses Groq's Orpheus TTS to instantly convert LLM text to a WAV audio stream."""
    try:
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="austin",  # Options: autumn (female), diana (female), hannah (female), austin (male), daniel (male), troy (male)
            input=text,
            response_format="wav"
        )
        return response.content
    except Exception as e:
        print(f"[TTS Error]: {e}")
        return b""

# --- FLASK ENDPOINTS ---
# Keep track of whether the assistant is actively engaged in a conversation session
active_sessions = {} # We can map this by an IP or keep a simple global flag if single-user

# Keep track of vision mode per user/device session
user_vision_modes = {} # In a real deployment, map this by a device ID header

# --- ENDPOINT 1: Dedicated lightweight wake-word checker ---
@app.route("/check-wake", methods=["POST"])
def check_wake_endpoint():
    if request.headers.get("X-Glasses-Secret") != GLASSES_SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        audio_file = request.files.get('audio_file')
        if not audio_file:
            return jsonify({"error": "No audio provided"}), 400

        audio_bytes = audio_file.read()

        # Transcribe short burst to check for "viper"
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3"
        )
        user_text = transcription.text.strip().lower()
        print(f"[Wake Check Heard]: {user_text}")

        if "viper" in user_text:
            print("[Server]: Wake word 'viper' detected! Signaling ESP32 to open mic.")
            return jsonify({"status": "awake"}), 200
        else:
            return Response(status=204) # Stay in passive listening mode

    except Exception as e:
        print(f"[Wake Check Error]: {e}")
        return jsonify({"error": str(e)}), 500


# --- ENDPOINT 2: Your exact existing chat & vision handler ---
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    global chat_history
    
    if request.headers.get("X-Glasses-Secret") != GLASSES_SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        audio_file = request.files.get('audio_file')
        image_file = request.files.get('image_file')

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        audio_bytes = audio_file.read()

        # 1. Transcribe via Groq Whisper
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3"
        )
        user_text = transcription.text.strip()
        print(f"[Glasses heard]: {user_text}")

        # (Your clean command & mode-switching logic continues here...)
        clean_command = user_text.lower().replace("viper", "").strip()

        mode_response_text = None
        if "start vision" in clean_command:
            print("[Server]: Switching session to VISION mode.")
            mode_response_text = "Vision active."
        elif "start voice" in clean_command or "stop vision" in clean_command:
            print("[Server]: Switching session to VOICE ONLY mode.")
            mode_response_text = "Voice only."

        if mode_response_text:
            wav_bytes = generate_tts_audio(mode_response_text)
            response_headers = {
                "X-Wake-Detected": "true",
                "X-Vision-Mode": "true" if "vision" in mode_response_text else "false"
            }
            return Response(wav_bytes, mimetype="audio/wav", headers=response_headers)

        # 3. Handle Normal Prompt + Optional Image
        if image_file:
            print("[Server]: Processing request WITH image attachment.")
            image_bytes = image_file.read()
            base64_img = base64.b64encode(image_bytes).decode('utf-8')
            
            chat_history = [{"role": "system", "content": SYSTEM_PROMPT}] 
            user_content = [
                {"type": "text", "text": clean_command if clean_command else "What am I looking at?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        else:
            user_content = clean_command if clean_command else "I'm listening."

        chat_history.append({"role": "user", "content": user_content})

        # 4. Groq LLM Generation & Tool Execution
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=chat_history,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=300
        )
        
        response_msg = response.choices[0].message
        
        if response_msg.tool_calls:
            chat_history.append(response_msg)
            for tool_call in response_msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                tool_result = execute_tool(tool_call.function.name, args)
                chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result
                })
            
            response = client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=chat_history,
                max_tokens=300
            )
            response_msg = response.choices[0].message

        final_text = response_msg.content
        chat_history.append({"role": "assistant", "content": final_text})
        print(f"[Assistant]: {final_text}")

        if len(chat_history) > 10:
            chat_history = [chat_history[0]] + chat_history[-6:]

        wav_bytes = generate_tts_audio(final_text)
        return Response(wav_bytes, mimetype="audio/wav", headers={"X-Wake-Detected": "true"})

    except Exception as e:
        print(f"[Error]: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
