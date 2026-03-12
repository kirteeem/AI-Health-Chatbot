# ===============================
# Load Environment Variables
# ===============================
from dotenv import load_dotenv
load_dotenv()

# ===============================
# Text To Speech (gTTS)
# ===============================
from gtts import gTTS


def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Convert doctor's text response into speech audio.

    Audio playback is handled automatically by Gradio,
    so we only generate and save the MP3 file here.
    """

    # Generate speech
    tts = gTTS(
        text=input_text,
        lang="en",
        slow=False
    )

    # Save audio file
    tts.save(output_filepath)

    # Return file path for Gradio audio output
    return output_filepath