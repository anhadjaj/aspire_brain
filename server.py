import os
import io
import time
import requests
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks
from fastapi.responses import Response
import uvicorn
from google import genai
from google.genai import types
import PIL.Image

# ================= CONFIGURATION & SECRETS =================
# We pull these from environment variables so they stay safe on Render
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_TTS_KEY = os.environ.get("GROQ_TTS_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
GOOGLE_MAPS_KEY = os.environ.get("GOOGLE_MAPS_KEY")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REFRESH_TOKEN = os.environ.get("SPOTIFY_REFRESH_TOKEN")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = os.environ.get("GOOGLE_REFRESH_TOKEN")

FIREBASE_DB_URL = "https://viper-assistant-267bb-default-rtdb.firebaseio.com"

# The security header your ESP32 must send
SECRET_GLASSES_TOKEN = os.environ.get("SECRET_GLASSES_TOKEN")

# ================= STATE =================
app = FastAPI(title="VIPER Gateway")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

spotify_token = {"access_token": "", "expires_at": 0}
google_token = {"access_token": "", "expires_at": 0}

# Store the latest image globally so the tools can access it
latest_image_bytes = None

# ================= OAUTH HELPERS =================
def get_spotify_token():
    if spotify_token["access_token"] and time.time() < spotify_token["expires_at"] - 300:
        return spotify_token["access_token"]
    res = requests.post("https://accounts.spotify.com/api/token", data={
        "grant_type": "refresh_token", "refresh_token": SPOTIFY_REFRESH_TOKEN,
        "client_id": SPOTIFY_CLIENT_ID, "client_secret": SPOTIFY_CLIENT_SECRET
    }).json()
    spotify_token["access_token"] = res.get("access_token", "")
    spotify_token["expires_at"] = time.time() + res.get("expires_in", 3600)
    return spotify_token["access_token"]

def get_google_token():
    if google_token["access_token"] and time.time() < google_token["expires_at"] - 300:
        return google_token["access_token"]
    res = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": GOOGLE_REFRESH_TOKEN, "grant_type": "refresh_token"
    }).json()
    google_token["access_token"] = res.get("access_token", "")
    google_token["expires_at"] = time.time() + res.get("expires_in", 3600)
    return google_token["access_token"]

# ================= TELEGRAM LOGGER =================
def send_telegram_log(user_text: str, ai_text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    msg = f"👤 You:\n{user_text}\n\n🤖 AI:\n{ai_text}"
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

# ================= VIPER TOOLS =================
def get_unread_emails(max_results: int = 3) -> str:
    """Fetch user's latest unread emails."""
    headers = {"Authorization": f"Bearer {get_google_token()}"}
    res = requests.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages?q=is:unread&maxResults={max_results}", headers=headers).json()
    if "messages" not in res: return "No unread emails."
    output = "Unread emails:\n"
    for msg in res["messages"]:
        detail = requests.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg['id']}?format=metadata&metadataHeaders=From&metadataHeaders=Subject", headers=headers).json()
        sender, subject = "Unknown", "No Subject"
        for header in detail.get("payload", {}).get("headers", []):
            if header["name"] == "From": sender = header["value"]
            if header["name"] == "Subject": subject = header["value"]
        output += f"- From: {sender} | Subject: {subject}\n"
    return output

def send_email(to_address: str, subject: str, body: str) -> str:
    """Send an email to a specific address."""
    import base64
    headers = {"Authorization": f"Bearer {get_google_token()}"}
    mime_msg = f"To: {to_address}\r\nSubject: {subject}\r\nContent-Type: text/plain; charset=\"UTF-8\"\r\n\r\n{body}"
    b64_msg = base64.urlsafe_b64encode(mime_msg.encode()).decode()
    res = requests.post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send", headers=headers, json={"raw": b64_msg})
    return f"Email sent to {to_address}." if res.status_code in [200, 201] else "Failed to send email."

def get_upcoming_events(max_results: int = 5) -> str:
    """Fetch upcoming schedule."""
    headers = {"Authorization": f"Bearer {get_google_token()}"}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = requests.get(f"https://www.googleapis.com/calendar/v3/calendars/primary/events?timeMin={now}&maxResults={max_results}&singleEvents=true&orderBy=startTime", headers=headers).json()
    if "items" not in res or not res["items"]: return "You have no upcoming events."
    output = "Upcoming events:\n"
    for item in res["items"]:
        start = item["start"].get("dateTime", item["start"].get("date", ""))
        output += f"- {item.get('summary', 'Untitled')} (Starts: {start})\n"
    return output

def create_event(summary: str, start_time: str, end_time: str = "") -> str:
    """Create a new calendar event. start_time MUST be ISO 8601."""
    headers = {"Authorization": f"Bearer {get_google_token()}"}
    if not end_time: end_time = start_time
    payload = {"summary": summary, "start": {"dateTime": start_time}, "end": {"dateTime": end_time}}
    res = requests.post("https://www.googleapis.com/calendar/v3/calendars/primary/events", headers=headers, json=payload)
    return "Event created successfully." if res.status_code in [200, 201] else "Failed to create event."

def search_drive(query: str) -> str:
    """Search Drive for a document."""
    headers = {"Authorization": f"Bearer {get_google_token()}"}
    q = query.replace(" ", "%20").replace("'", "%27")
    res = requests.get(f"https://www.googleapis.com/drive/v3/files?q=name%20contains%20%27{q}%27&pageSize=1&fields=files(id,name,mimeType)", headers=headers).json()
    files = res.get("files", [])
    if not files: return "No matching file found."
    
    file_id, name, mime = files[0]["id"], files[0]["name"], files[0]["mimeType"]
    fetch_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=text/plain" if "document" in mime else f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    
    text_res = requests.get(fetch_url, headers=headers)
    if text_res.status_code == 200:
        text = text_res.text[:1500] + "...[Truncated]" if len(text_res.text) > 1500 else text_res.text
        return f"Content of '{name}':\n{text}"
    return f"Found '{name}', but cannot read its contents directly."

def take_photo() -> str:
    """Take a high-resolution photo and send it via Telegram."""
    if not latest_image_bytes: return "Camera failed to capture an image."
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": ("snap.jpg", latest_image_bytes, "image/jpeg")})
    return "Photo successfully sent to your phone."

def get_weather(lat: float, lng: float) -> str:
    """Gets current weather conditions."""
    res = requests.get(f"https://weather.googleapis.com/v1/currentConditions:lookup?key={GOOGLE_MAPS_KEY}&location.latitude={lat}&location.longitude={lng}&unitsSystem=METRIC").json()
    temp = res.get("temperature", {}).get("degrees", 0.0)
    cond = res.get("weatherCondition", {}).get("description", {}).get("text", "Unknown")
    return f"Weather is {cond}, {temp} degrees Celsius."

def control_spotify(action: str, query: str = "", volume: int = 50) -> str:
    """Control Spotify playback."""
    headers = {"Authorization": f"Bearer {get_spotify_token()}"}
    action = action.lower()
    
    if action in ["status", "currently_playing"]:
        res = requests.get("https://api.spotify.com/v1/me/player/currently-playing", headers=headers)
        if res.status_code == 204: return "Music is currently paused."
        data = res.json()
        if data.get("is_playing"): return f"Playing {data['item']['name']} by {data['item']['artists'][0]['name']}."
        return "Music is paused."
        
    if action == "play_track" or (action == "play" and query):
        res = requests.get(f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1", headers=headers).json()
        tracks = res.get("tracks", {}).get("items", [])
        if not tracks: return "Couldn't find that track."
        requests.put("https://api.spotify.com/v1/me/player/play", headers=headers, json={"uris": [tracks[0]["uri"]]})
        return f"Playing {tracks[0]['name']}."
        
    url_map = {"pause": "pause", "play": "play", "resume": "play", "next": "next", "previous": "previous"}
    if action in url_map:
        if action in ["next", "previous"]: requests.post(f"https://api.spotify.com/v1/me/player/{url_map[action]}", headers=headers)
        else: requests.put(f"https://api.spotify.com/v1/me/player/{url_map[action]}", headers=headers)
        return "Command executed."
    return "Unsupported Spotify action."

def get_current_location() -> str:
    """Returns the user's current context location."""
    # Since the server can't scan local WiFi, we return a mocked context point to fulfill the tool requirement
    return "Current Location: Botanical Garden metro station, Noida, Uttar Pradesh, India."

def get_directions(origin: str, destination: str, mode: str = "driving") -> str:
    """Get travel time, traffic ETA, and distance."""
    res = requests.get(f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&mode={mode}&departure_time=now&key={GOOGLE_MAPS_KEY}").json()
    if not res.get("routes"): return "Route not found."
    leg = res["routes"][0]["legs"][0]
    return f"Distance: {leg['distance']['text']}. Standard ETA: {leg['duration']['text']}. Live ETA with Traffic: {leg.get('duration_in_traffic', leg['duration'])['text']}."

def search_places(query: str) -> str:
    """Search for places or restaurants near a location."""
    res = requests.get(f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&key={GOOGLE_MAPS_KEY}").json()
    results = res.get("results", [])
    if not results: return "No places found."
    output = "Top results: "
    for r in results[:3]:
        output += f"{r.get('name')} (Rating: {r.get('rating', 0)}). "
    return output

def google_search(query: str) -> str:
    """Searches the live internet via Tavily."""
    res = requests.post("https://api.tavily.com/search", headers={"Authorization": f"Bearer {TAVILY_API_KEY}"}, json={"query": query, "search_depth": "basic", "include_answer": True}).json()
    return res.get("answer", res.get("results", [{"content": "No results found."}])[0]["content"])

def get_news(category: str = "general") -> str:
    """Fetches breaking global headlines."""
    res = requests.get(f"https://newsapi.org/v2/top-headlines?category={category}&language=en&pageSize=3&apiKey={NEWS_API_KEY}").json()
    if not res.get("articles"): return "No news found."
    return "Headlines: " + " ".join([f"{i+1}. {a['title']}." for i, a in enumerate(res["articles"])])

def analyze_camera_frame(question: str) -> str:
    """Capture a photo and analyze it to answer the user's visual question."""
    if not latest_image_bytes: return "Error: Camera failed to capture an image."
    return "I have attached the image to our conversation context. Please look at it and answer the user's question directly."

def get_recipe_step(recipe_name: str, step_number: int) -> str:
    """Fetch a specific recipe step from Firebase."""
    res = requests.get(f"{FIREBASE_DB_URL}/recipes/{recipe_name}/step_{step_number}.json")
    if res.status_code == 200 and res.json() is not None:
        return f"Step {step_number}: {res.json()}"
    return "You have completed all the steps for this recipe!"

def save_memory(note: str) -> str:
    """Save a fact to the Firebase memory database."""
    res = requests.post(f"{FIREBASE_DB_URL}/memories.json", json={"note": note})
    return "Memory successfully saved." if res.status_code == 200 else "Failed to save memory."

def fetch_memories() -> str:
    """Retrieve all saved notes from Firebase."""
    res = requests.get(f"{FIREBASE_DB_URL}/memories.json").json()
    if not res: return "Your memory bank is empty."
    output = "Saved memories:\n"
    for v in res.values():
        if isinstance(v, dict) and "note" in v: output += f"- {v['note']}\n"
    return output

def clear_memories() -> str:
    """Delete all saved memories in Firebase."""
    requests.delete(f"{FIREBASE_DB_URL}/memories.json")
    return "All memories have been erased."

# Gemini needs to know what tools exist
VIPER_TOOLS = [
    get_unread_emails, send_email, get_upcoming_events, create_event, search_drive,
    take_photo, get_weather, control_spotify, get_current_location, get_directions,
    search_places, google_search, get_news, analyze_camera_frame,
    get_recipe_step, save_memory, fetch_memories, clear_memories
]

# ================= ENDPOINTS =================
@app.post("/wake")
async def check_wake_word(audio: UploadFile = File(...), x_glasses_secret: str = Header(None)):
    """Fast STT to check for the wake word."""
    if x_glasses_secret != SECRET_GLASSES_TOKEN: raise HTTPException(status_code=403, detail="Unauthorized")
    
    audio_bytes = await audio.read()
    res = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions", 
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, 
        files={"file": ("wake.wav", audio_bytes, "audio/wav")}, 
        data={"model": "whisper-large-v3-turbo", "response_format": "json"}
    )
    text = res.json().get("text", "").lower()
    return {"status": "awake"} if "viper" in text or "vyper" in text else {"status": "sleep"}

@app.post("/chat")
async def chat_turn(background_tasks: BackgroundTasks, audio: UploadFile = File(...), image: UploadFile = File(None), x_glasses_secret: str = Header(None)):
    """The main pipeline: Transcribe -> Process with Tools -> Generate TTS"""
    global latest_image_bytes
    if x_glasses_secret != SECRET_GLASSES_TOKEN: raise HTTPException(status_code=403, detail="Unauthorized")
    
    # 1. Transcribe Voice
    audio_bytes = await audio.read()
    res = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions", 
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, 
        files={"file": ("chat.wav", audio_bytes, "audio/wav")}, 
        data={"model": "whisper-large-v3-turbo", "response_format": "json"}
    )
    user_text = res.json().get("text", "").strip()
    if not user_text: return Response(content=b"", media_type="audio/wav")
    print(f"\n[You]: {user_text}")

    # 2. Extract Camera Frame
    contents = [user_text]
    if image:
        latest_image_bytes = await image.read()
        contents.append(PIL.Image.open(io.BytesIO(latest_image_bytes)))

    # 3. Ask Gemini (Automatically handles tool loops!)
    sys_prompt = (
        "You are VIPER, a concise conversational AI assistant for Anhad's smart glasses. "
        "Speak naturally with no markdown, asterisks, or emojis. Output is spoken aloud. Keep answers brief. "
        f"Today is {time.strftime('%A, %B %d, %Y')}. Assume the camera frame is rotated 90° counter-clockwise."
    )
    
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=sys_prompt,
            tools=VIPER_TOOLS,
            temperature=0.1,
        )
    )
    
    ai_text = response.text.strip() if response.text else "Done."
    print(f"[VIPER]: {ai_text}")
    
    # Log to Telegram in the background so it doesn't slow down the audio return
    background_tasks.add_task(send_telegram_log, user_text, ai_text)

    # 4. Generate TTS via Groq
    tts_res = requests.post(
        "https://api.groq.com/openai/v1/audio/speech", 
        headers={"Authorization": f"Bearer {GROQ_TTS_KEY}"}, 
        json={"model": "canopylabs/orpheus-v1-english", "voice": "troy", "response_format": "wav", "input": ai_text}
    )
    
    return Response(content=tts_res.content, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
