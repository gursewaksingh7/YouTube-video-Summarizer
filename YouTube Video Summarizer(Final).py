import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import whisper
import re
import subprocess

# Summarizers
@st.cache_resource
def load_english_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_hindi_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# Toxicity
@st.cache_resource
def load_toxicity_classifier():
    return pipeline(
        "text-classification",
        model="unitary/multilingual-toxic-xlm-roberta",
        tokenizer="unitary/multilingual-toxic-xlm-roberta"
    )

# Zero-shot misinformation
@st.cache_resource
def load_misinformation_detector():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

# Whisper
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Initialize pipelines
english_summarizer = load_english_summarizer()
hindi_summarizer = load_hindi_summarizer()
toxicity_classifier = load_toxicity_classifier()
misinformation_detector = load_misinformation_detector()
whisper_model = load_whisper()

# Hindi bad words
HINDI_BAD_WORDS = [
    "chutiya", "madarchod", "bhenchod", "gandu", "bhosdi", "lund", "gaand",
    "haraami", "kutte", "saala", "haraamkhor", "randi", "behen ke laude", "madarchod ke bachhe"
]

def is_hindi(text):
    hindi_chars = re.findall(r"[\u0900-\u097F]", text)
    return len(hindi_chars) > 50

def is_toxic_strict(text, chunk_size=512, threshold=0.15):
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        preds = toxicity_classifier(chunk)
        for pred in preds:
            if pred["label"] == "toxic" and pred["score"] > threshold:
                return True
    return False

def keyword_toxicity(text):
    text_lower = text.lower()
    return any(word in text_lower for word in HINDI_BAD_WORDS)

# fast misinformation check
def is_misinformation(text, threshold=0.8):
    sample = text[:3000]  # only check first 3000 characters
    labels = ["misinformation", "factual"]
    preds = misinformation_detector(sample, candidate_labels=labels)
    misinfo_index = preds["labels"].index("misinformation")
    misinfo_score = preds["scores"][misinfo_index]
    return misinfo_score

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

# summarizers
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

def summarize_hindi(text, chunk_size_words=250):
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
st.set_page_config(page_title="YouTube Summarizer Advanced", layout="centered")
st.title("YouTube Summarizer (Search + Improved Quality)")

youtube_link = st.text_input("Enter YouTube video link:")

if youtube_link:
    transcript, video_id = extract_transcript(youtube_link)

    if video_id:
        st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

    if not transcript:
        with st.spinner("No captions found — downloading audio and transcribing..."):
            try:
                audio_file = download_audio(youtube_link)
                transcript = transcribe_audio(audio_file)
            except Exception as e:
                st.error(f"Audio download or transcription failed: {e}")
                st.stop()

    st.write(f"Transcript length: {len(transcript)} characters")
    st.text_area("Transcript Preview (first 1000 chars):", transcript[:1000])

    # keyword search
    search_keyword = st.text_input("Search within Transcript:")
    if search_keyword:
        matches = list(re.finditer(re.escape(search_keyword), transcript, re.IGNORECASE))
        st.info(f"Found {len(matches)} occurrences of **{search_keyword}** in transcript.")
        highlighted_preview = re.sub(
            f"({re.escape(search_keyword)})",
            r"**\1**",
            transcript[:3000],
            flags=re.IGNORECASE
        )
        st.markdown(highlighted_preview)

    if st.button("Generate Summary"):
        with st.spinner("Checking for abusive language..."):
            toxic_flag = is_toxic_strict(transcript, threshold=0.15)
            keyword_flag = keyword_toxicity(transcript)

            if toxic_flag or keyword_flag:
                st.error("This video transcript contains abusive or inappropriate language. Summarization has been skipped.")
                st.stop()

        with st.spinner("Checking for misinformation..."):
            misinfo_score = is_misinformation(transcript, threshold=0.8)
            st.info(f"Misinformation confidence score (first 3000 chars): {misinfo_score:.2f}")
            misinfo_flag = misinfo_score > 0.8
            if misinfo_flag:
                st.warning("This video appears to contain potential misinformation.")

        with st.spinner("Summarizing..."):
            if is_hindi(transcript):
                st.info("Detected Hindi transcript — summarizing with mT5 XLSum...")
                summary = summarize_hindi(transcript)
            else:
                st.info("Detected English transcript — summarizing with Pegasus XSum...")
                summary = summarize_english(transcript)

            if misinfo_flag:
                summary = "*Potential misinformation detected.*\n\n" + summary

        if summary:
            st.subheader("Final Summary")
            st.write(summary)
            st.download_button("Download Summary", summary, file_name="summary.txt")
        else:
            st.warning("Could not generate a summary.")
