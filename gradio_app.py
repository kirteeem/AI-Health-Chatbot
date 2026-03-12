# ===============================
# Load Environment Variables
# ===============================
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import os
from groq import Groq

from brain import analyze_image_with_query
from patient_query import transcribe_with_groq
from doctor_response import text_to_speech_with_gtts


# ===============================
# GROQ Client
# ===============================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ===============================
# Doctor Prompt
# ===============================
system_prompt = """
You are a professional doctor.
Provide short medical guidance based on symptoms.
Maximum two sentences.
"""


# ===============================
# Voice Only Doctor Response
# ===============================
def symptom_based_response(user_text):

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": system_prompt + " Symptoms: " + user_text
            }
        ],
    )

    return response.choices[0].message.content


# ===============================
# Main Function
# ===============================
def process_inputs(audio_filepath, image_filepath):

    # 🎤 Speech → Text
    speech_to_text_output = transcribe_with_groq(
        "whisper-large-v3",
        audio_filepath
    )

    # 🧠 Logic
    if image_filepath is not None:
        doctor_response = analyze_image_with_query(
            system_prompt + speech_to_text_output,
            image_filepath
        )
    else:
        doctor_response = symptom_based_response(
            speech_to_text_output
        )

    # 🔊 Voice Output
    audio_path = text_to_speech_with_gtts(
        doctor_response,
        "final.mp3"
    )

    return speech_to_text_output, doctor_response, audio_path


# ===============================
# UI
# ===============================
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath", label="Optional Image")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor Response"),
        gr.Audio(type="filepath", label="Doctor Voice")
    ],
    title="AI Doctor - Vision & Voice"
)


iface.queue()

port = int(os.environ.get("PORT", 7860))

iface.launch(
    server_name="0.0.0.0",
    server_port=port,
    debug=True
)
