import speech_recognition as sr
from gtts import gTTS
import os
from transformers import pipeline
import time

# Initialize Speech-to-Text Recognizer
recognizer = sr.Recognizer()

def speech_to_text(audio_file, retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        try:
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                print("Recognized Text:", text)
                return text
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Request failed: {e}. Retrying ({attempt+1}/{retries})...")
            attempt += 1
            time.sleep(delay)  # Wait before retrying
        except Exception as e:
            print(f"Unexpected error: {e}.")
            return ""
    
    print("Failed to recognize speech after several attempts.")
    return ""

# Initialize Text-to-Speech
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    os.system(f"start {output_file}")  # Adjust for your OS (e.g., `open` on MacOS, `start` on Windows)

# Initialize Hugging Face Model for Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example Workflow
audio_input = "C:\Users\mi\Downloads\audio.mp3.wav"  # Replace with the actual path to your audio file
recognized_text = speech_to_text(audio_input)

if recognized_text:
    # Summarizing the recognized text
    summary = summarizer(recognized_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    print("Summary:", summary)

    # Convert the summary back to speech
    text_to_speech(summary, "summary_audio.mp3")
