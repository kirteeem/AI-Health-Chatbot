# ===============================
# Load Environment Variables
# ===============================
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from pydub import AudioSegment
from io import BytesIO
from groq import Groq


# ===============================
# Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ===============================
# Record Audio
# ===============================
def record_audio(file_path, timeout=20, phrase_time_limit=None):

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            logging.info("Start speaking...")
            audio_data = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )

            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")

            logging.info(f"Saved audio → {file_path}")

    except Exception as e:
        logging.error(f"Recording error: {e}")


# ===============================
# Speech → Text using GROQ
# ===============================
def transcribe_with_groq(stt_model, audio_filepath):

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY missing in .env")

    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text
