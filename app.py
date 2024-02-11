import os
import logging
import httpx
import numpy as np
import pennylane as qml
import re
import json
from textblob import TextBlob
import aiosqlite
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
import whisper
import time

nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Read OpenAI API key from config.json
with open('config.json') as config_file:
    config_data = json.load(config_file)
    OPENAI_API_KEY = config_data.get('openai_api_key', None)

if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found in config.json. Please provide a valid API key.")

# Configuration
GPT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# Get the current working directory
current_directory = os.getcwd()

# Assume your audio files are in the same directory as the script
AUDIO_PATHS = [file for file in os.listdir(current_directory) if file.endswith(".mp3")]

# Check if no MP3 files are found
if not AUDIO_PATHS:
    logging.error("Error: No MP3 files found in the specified directory. Please make sure there are MP3 files present.")
    exit()

async def init_db():
    with ThreadPoolExecutor() as pool:
        await asyncio.get_event_loop().run_in_executor(pool, _init_db)

def _init_db():
    with aiosqlite.connect('colobits.db') as db:
        with db.cursor() as cursor:
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS analysis_table (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    color_code TEXT,
                    quantum_state TEXT,
                    reply TEXT,
                    report TEXT
                )
                '''
            )
            db.commit()
    logging.debug("Database initialization complete.")

def save_to_database(color_code, quantum_state, reply, report):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(pool, _save_to_database, color_code, quantum_state, reply, report)

def _save_to_database(color_code, quantum_state, reply, report):
    with aiosqlite.connect('colobits.db') as db:
        with db.cursor() as cursor:
            cursor.execute(
                "INSERT INTO analysis_table (color_code, quantum_state, reply, report) VALUES (?, ?, ?, ?)",
                (color_code, quantum_state, reply, report)
            )
            db.commit()
    logging.debug("Data saved to the database.")

@qml.qnode(qml.device("default.qubit", wires=4))
def quantum_circuit(color_code, amplitude):
    logging.debug(f"Running quantum circuit with color_code: {color_code}, amplitude: {amplitude}")
    r, g, b = (int(color_code[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    qml.RY(r * np.pi, wires=0)
    qml.RY(g * np.pi, wires=1)
    qml.RY(b * np.pi, wires=2)
    qml.RY(amplitude * np.pi, wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    return qml.probs(wires=[0, 1, 2, 3])

def sentiment_to_amplitude(text):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(pool, _sentiment_to_amplitude, text)

def _sentiment_to_amplitude(text):
    logging.debug("Analyzing sentiment...")
    analysis = TextBlob(text)
    return (analysis.sentiment.polarity + 1) / 2

def extract_color_code(response_text):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(pool, _extract_color_code, response_text)

def _extract_color_code(response_text):
    logging.debug("Extracting color code...")
    pattern = r'#([0-9a-fA-F]{3,6})'
    match = re.search(pattern, response_text)
    if match:
        color_code = match.group(1)
        if len(color_code) == 3:
            color_code = ''.join([char*2 for char in color_code])
        return color_code
    return None

def generate_html_color_codes(sentence, attempt=0):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(pool, _generate_html_color_codes, sentence, attempt)

def _generate_html_color_codes(sentence, attempt):
    logging.debug("Generating HTML color codes...")
    max_retries = 4
    retry_delay = 1

    prompts = [
        f"Generate an HTML color code that represents the emotion of an AT&T employee discussing corporate practices. Summary: '{sentence}'",
        f"Suggest a color code reflecting the mood of an AT&T employee when considering corporate practices. Summary: '{sentence}'?",
        f"Provide a color that matches the sentiment of an AT&T employee discussing corporate practices. Summary: '{sentence}'.",
        f"What color best captures the feelings of an AT&T employee and their views on corporate practices? Summary: '{sentence}'?"
    ]

    while attempt < max_retries:
        prompt = prompts[attempt]

        response = call_openai_api(prompt)
        if response.status_code == 429:
            logging.warning("Rate limit reached, will retry after delay.")
            time.sleep(retry_delay)
            retry_delay *= 2
            continue

        if response.status_code != 200:
            logging.error(f"OpenAI API error with status code {response.status_code}: {response.text}")
            return None

        response_text = response.json()['choices'][0]['message']['content'].strip()
        color_code = extract_color_code(response_text)
        if color_code:
            logging.info(f"Color code extracted: {color_code}")
            return color_code
        else:
            logging.warning(f"No valid color code found in response. Retrying with a different prompt.")
            attempt += 1
            time.sleep(retry_delay)

    return None

def call_openai_api(prompt, max_retries=3, retry_delay=15):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(pool, _call_openai_api, prompt, max_retries, retry_delay)

def _call_openai_api(prompt, max_retries, retry_delay):
    logging.debug(f"Calling OpenAI API with prompt: {prompt}")
    attempt = 0
    while attempt < max_retries:
        logging.debug(f"OpenAI API call attempt {attempt + 1} with prompt: {prompt}")
        try:
            with httpx.Client(timeout=30.0) as client:  # Increased timeout
                data = {
                    "model": GPT_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS
                }
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                response = client.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
                response.raise_for_status()  # Raises an exception for 4XX/5XX responses
                logging.debug(f"OpenAI API response: {response.json()}")
                return response
        except (httpx.ReadTimeout, httpx.SSLWantReadError):
            logging.warning(f"Request attempt {attempt + 1} failed. Retrying after {retry_delay} seconds...")
            time.sleep(retry_delay)
            attempt += 1
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break  # Exit the loop if an unexpected error occurs

    return None

def whisper_integration(audio_path):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(pool, _whisper_integration, audio_path)

def _whisper_integration(audio_path):
    logging.debug("Integrating Whisper ASR...")
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_path)
    if audio is None:
        return None, None  # Handle the case where loading audio fails
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    recognized_text = result.text

    # Mask sensitive information in recognized text
    recognized_text = re.sub(r'\b\d{4}\b', '****', recognized_text)  # Mask 4-digit pins
    recognized_text = re.sub(r'\b\d{5}-\d{4}\b', '*****-****', recognized_text)  # Mask ZIP+4 codes
    recognized_text = re.sub(r'\b\d{5}\b', '*****', recognized_text)  # Mask 5-digit ZIP codes
    recognized_text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****', recognized_text)  # Mask SSN-like patterns

    return detected_language, recognized_text

def create_audio_conversations_json():
    conversations = []

    for audio_path in AUDIO_PATHS:
        transcript_path = f"{os.path.splitext(audio_path)[0]}_transcript.txt"
        detected_language, recognized_text = whisper_integration(audio_path)

        conversation = {
            "transcript_path": transcript_path,
            "audio_path": audio_path,
            "detected_language": detected_language,
            "recognized_text": recognized_text
        }

        conversations.append(conversation)

    with open('audio_conversations.json', 'w') as json_file:
        json.dump(conversations, json_file, indent=2)

def process_user_input(transcript_path, audio_path):
    with open(transcript_path, 'r') as file:
        user_input = file.read()

    sentiment_amplitude = sentiment_to_amplitude(user_input)
    color_code = generate_html_color_codes(user_input)
    if not color_code:
        color_code = "Error"

    amplitude = sentiment_to_amplitude(user_input)
    quantum_state = quantum_circuit(color_code, amplitude) if color_code != "Error" else "Error"

    # Whisper ASR integration
    detected_language, recognized_text = whisper_integration(audio_path)

    return {
        "user_input": user_input,
        "sentiment_amplitude": sentiment_amplitude,
        "color_code": color_code,
        "quantum_state": quantum_state,
        "detected_language": detected_language,
        "recognized_text": recognized_text
    }

def process_conversations(conversations):
    with ThreadPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(pool, process_user_input, conversation["transcript_path"], conversation["audio_path"]) for conversation in conversations]
        results = loop.run_until_complete(asyncio.gather(*tasks))
    return results

def main():
    logging.debug("Starting main process...")
    init_db()
    create_audio_conversations_json()

    markdown_content = "# Analysis Report\n\n"

    with open('audio_conversations.json') as json_file:
        conversations = json.load(json_file)

    results = process_conversations(conversations)

    for result in results:
        logging.debug(f"Processing user input: {result['user_input']}")

        markdown_content += f"## User Input\n\n- **Input**:\n\n{result['user_input']}\n\n"
        markdown_content += f"## Sentiment Amplitude\n\n- **Amplitude**: {result['sentiment_amplitude']}\n\n"
        markdown_content += f"## HTML Color Code\n\n- **Color Code**: {result['color_code']}\n\n"
        markdown_content += f"## Quantum State: {result['quantum_state']}\n"
        markdown_content += f"## Detected Language: {result['detected_language']}\n"
        markdown_content += f"## Recognized Text (Whisper ASR): {result['recognized_text']}\n\n"

    print(markdown_content)
    logging.debug("Main process complete.")

if __name__ == "__main__":
    main()
