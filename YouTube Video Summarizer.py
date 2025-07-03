import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import whisper
import re
import subprocess

# English summarizer (distilbart)
@st.cache_resource
def load_english_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# Hindi summarizer (mT5 XLSum) without src_lang/tgt_lang
@st.cache_resource
def load_hindi_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# Whisper
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

english_summarizer = load_english_summarizer()
hindi_summarizer = load_hindi_summarizer()
whisper_model = load_whisper()

def is_hindi(text):
    hindi_chars = re.findall(r"[\u0900-\u097F]", text)
    return len(hindi_chars) > 50

def extract_transcript(youtube_video_url):
    try:
        video_id = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", youtube_video_url).group(1)
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en"])
        transcript = " ".join([i["text"] for i in transcript_data])
        return transcript, video_id
    except Exception as e:
        st.warning(f"No captions found: {e}")
        return None, None

def download_audio(youtube_url, filename="audio.mp3"):
    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "-o", filename,
        youtube_url,
    ]
    subprocess.run(command, check=True)
    return filename

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def summarize_english(text, chunk_size_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk = " ".join(words[i:i+chunk_size_words])
        chunks.append(chunk)

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunk.split()) < 10:
            continue
        try:
            summary = english_summarizer(chunk, max_length=300, min_length=80, do_sample=False)
            summaries.append(summary[0]["summary_text"])
        except Exception as e:
            st.warning(f"English chunk {i} failed: {e}")
    return "\n\n".join(summaries)

def summarize_hindi(text, chunk_size_words=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk = " ".join(words[i:i+chunk_size_words])
        chunks.append(chunk)

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunk.split()) < 10:
            continue
        try:
            summary = hindi_summarizer(chunk, max_length=300, min_length=80, do_sample=False)
            summaries.append(summary[0]["summary_text"])
        except Exception as e:
            st.warning(f"Hindi chunk {i} failed: {e}")
    return "\n\n".join(summaries)

# Streamlit UI
st.set_page_config(page_title="Hindi + English Summarizer", layout="centered")
st.title("YouTube Summarizer (Hindi + English)")

youtube_link = st.text_input("Enter YouTube video link:")

if youtube_link:
    transcript, video_id = extract_transcript(youtube_link)

    if video_id:
        st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

    if not transcript:
        with st.spinner("No captions — downloading audio and transcribing..."):
            try:
                audio_file = download_audio(youtube_link)
                transcript = transcribe_audio(audio_file)
            except Exception as e:
                st.error(f"Audio download/transcription failed: {e}")
                st.stop()

    st.write(f"Transcript length: {len(transcript)} characters")
    st.text_area("Transcript Preview (first 1000 chars):", transcript[:1000])

    if st.button("Generate Summary"):
        if is_hindi(transcript):
            st.info("Detected Hindi transcript — summarizing with mT5 XLSum...")
            with st.spinner("Summarizing Hindi..."):
                summary = summarize_hindi(transcript)
        else:
            st.info("Detected English transcript — summarizing with distilbart...")
            with st.spinner("Summarizing English..."):
                summary = summarize_english(transcript)

        if summary:
            st.subheader("Final Summary")
            st.write(summary)
            st.download_button("Download Summary", summary, file_name="summary.txt")
        else:
            st.warning("Could not generate a summary.")
