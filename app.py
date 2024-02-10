
import asyncio
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
from google.colab import userdata

nest_asyncio.apply()

OPENAI_API_KEY = userdata.get('openai_api_key')
executor = ThreadPoolExecutor()

GPT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.7

async def init_db():
    async with aiosqlite.connect('colobits.db') as db:
        async with db.cursor() as cursor:
            await cursor.execute(
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
            await db.commit()

async def save_to_database(color_code, quantum_state, reply, report):
    async with aiosqlite.connect('colobits.db') as db:
        async with db.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO analysis_table (color_code, quantum_state, reply, report) VALUES (?, ?, ?, ?)",
                (color_code, quantum_state, reply, report)
            )
            await db.commit()

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

async def sentiment_to_amplitude(text):
    analysis = TextBlob(text)
    return (analysis.sentiment.polarity + 1) / 2

def extract_color_code(response_text):
    pattern = r'#([0-9a-fA-F]{3,6})'
    match = re.search(pattern, response_text)
    if match:
        color_code = match.group(1)
        if len(color_code) == 3:
            color_code = ''.join([char*2 for char in color_code])
        return color_code
    return None

async def generate_html_color_codes(sentence, attempt=0):
    max_retries = 4
    retry_delay = 1

    prompts = [
        f"Please generate an HTML color code that best represents the emotion of an AT&T employee and their thoughts on corporate practices. Summary: '{sentence}'",
        f"Suggest a color code reflecting the mood of an AT&T employee when considering corporate practices. Summary: '{sentence}'?",
        f"I need a color that matches the sentiment of an AT&T employee discussing corporate practices. Summary: '{sentence}'.",
        f"What color best captures the feelings of an AT&T employee and their views on corporate practices? Summary: '{sentence}'?"
    ]

    while attempt < max_retries:
        prompt = prompts[attempt]

        response = await call_openai_api(prompt)
        if response.status_code == 429:
            logging.warning("Rate limit reached, will retry after delay.")
            await asyncio.sleep(retry_delay)
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
            await asyncio.sleep(retry_delay)

    return None

async def call_openai_api(prompt, max_retries=3, retry_delay=15):
    attempt = 0
    while attempt < max_retries:
        logging.debug(f"OpenAI API call attempt {attempt + 1} with prompt: {prompt}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
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
                response = await client.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
                response.raise_for_status()  # Raises an exception for 4XX/5XX responses
                return response
        except (httpx.ReadTimeout, httpx.SSLWantReadError):
            logging.warning(f"Request attempt {attempt + 1} failed. Retrying after {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            attempt += 1
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break  # Exit the loop if an unexpected error occurs
    logging.debug("All attempts failed or an unexpected error occurred")
    return None

async def whisper_integration(audio_path):
    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    recognized_text = result.text
    return detected_language, recognized_text

async def process_user_input(user_input, audio_path):
    sentiment_amplitude = await sentiment_to_amplitude(user_input)
    color_code = await generate_html_color_codes(user_input)
    if not color_code:
        color_code = "Error"

    amplitude = await sentiment_to_amplitude(user_input)
    quantum_state = quantum_circuit(color_code, amplitude) if color_code != "Error" else "Error"

    # Whisper ASR integration
    detected_language, recognized_text = await whisper_integration(audio_path)

    advanced_interaction = await advanced_gpt4_interaction(user_input, quantum_state, color_code)

    markdown_output = f"## User Input\n\n- **Input**:

 {user_input}\n\n"
    markdown_output += f"## Sentiment Amplitude\n\n- **Amplitude**: {sentiment_amplitude}\n\n"
    markdown_output += f"## HTML Color Code\n\n- **Color Code**: {color_code}\n\n"
    markdown_output += f"## Quantum State: {quantum_state}\n"
    markdown_output += f"## Detected Language: {detected_language}\n"
    markdown_output += f"## Recognized Text (Whisper ASR): {recognized_text}\n"
    markdown_output += f"## Advanced GPT-4 Interaction\n\n{advanced_interaction}\n\n"

    return markdown_output

def generate_rejection_report(conversation_text, quantum_state, color_code):
    return f"Rejection Report:\nBased on the analysis, the following reasons were identified for rejection:\n{conversation_text}"

def generate_approval_report(conversation_text, quantum_state, color_code):
    return f"Approval Report:\nBased on the analysis, the interaction meets the criteria for approval:\n{conversation_text}"

async def advanced_gpt4_interaction(conversation_text, quantum_state, color_code):
    max_retries = 3
    retry_delay = 1
    pause_between_requests = 3

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                prompt = (
                    "This is a conversation between a host and a guest from an Airbnb-like app. "
                    "Analyze the conversation for risk factors and compassion aspects, considering the quantum states and HTML color codes.\n\n"
                    f"Conversation:\n{conversation_text}\n\n"
                    f"Quantum State: {quantum_state}\n"
                    f"HTML Color Code: {color_code}\n\n"
                    "Agent 1 (Risk Assessment AI): [Analyzes the conversation for risk factors]\n"
                    "Agent 2 (Compassionate Decision-Making AI): [Considers the compassion aspects of the interaction]\n\n"
                    "The analysis should follow these guidelines: 1. Objective analysis, 2. Ethical considerations, 3. Emotional intelligence, 4. Practical advice.\n\n"
                    "Begin the analysis: [Reply] [decision] either accept or deny the guest's safety stay consideration with the report, making sure to include either accept or deny in the report. [/Reply] [/decision]"
                )

                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are two advanced AI systems."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 300,
                }
                headers = {"Authorization": f"Bearer {userdata.get('openai_api_key')}"}
                response = await client.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)

                if response.status_code == 429:
                    logging.warning("Rate limit reached, will retry after delay.")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue

                if response.status_code != 200:
                    logging.error(f"OpenAI API error with status code {response.status_code}: {response.text}")
                    return f"Error in OpenAI API call: Status code {response.status_code}"

                response_text = response.json()['choices'][0]['message']['content'].strip()

                if re.search(r"\b(reject|deny|decline)\b", response_text, re.IGNORECASE):
                    report = generate_rejection_report(conversation_text, quantum_state, color_code)
                    await save_to_database(color_code, quantum_state, "deny", report)
                    return report
                elif re.search(r"\b(approve|accept|confirm)\b", response_text, re.IGNORECASE):
                    report = generate_approval_report(conversation_text, quantum_state, color_code)
                    await save_to_database(color_code, quantum_state, "accept", report)
                    return report
                else:
                    return "Decision could not be determined from the analysis."

        except httpx.RequestError as req_err:
            logging.error(f"Request error occurred: {req_err}")
            if attempt < max_retries - 1:
                logging.info("Retrying...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                return "Error: Request error in OpenAI API call."

async def process_conversations(conversations):
    results = []
    for conversation in conversations:
        result = await process_user_input(conversation["user_input"], conversation["audio_path"])
        results.append(result)
    return results

async def main():
    await init_db()
    markdown_content = "# Analysis Report\n\n"

    with open('synthetic_conversations.json', 'r') as file:
        data = json.load(file)

    for category, conversations in data.items():
        markdown_content += f"## {category.title()}\n\n"
        results = await process_conversations(conversations)
        for result in results:
            markdown_content += result

    print(markdown_content)

if __name__ == "__main__":
    asyncio.run(main())
```

This code includes the modifications to the prompts in the `generate_html_color_codes` function and maintains the integration with the Whisper ASR system.
