YouTube Summarizer (Multilingual + Toxicity & Misinformation Detection)

This is a powerful Streamlit web application that summarizes YouTube videos in English and Hindi, while also detecting toxic language and misinformation

# Features

- Summarizes YouTube videos with or without captions
- Supports both Hindi and English transcripts
- Filters out videos with abusive/toxic language
- Warns about possible misinformation
- Uses OpenAI Whisper to transcribe audio if captions are not available
- Built-in keyword search in transcripts
- Option to download the final summary

# Tech Stack

- [Streamlit](https://streamlit.io/) – UI framework
- [transformers (HuggingFace)](https://huggingface.co/) – For summarization & classification
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) – Fetch captions
- [OpenAI Whisper](https://github.com/openai/whisper) – Audio transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) – Download YouTube audio

# Models Used

[Task]                        [Model Name]                                         
English Summarization    -  `google/pegasus-xsum`                            
Hindi Summarization      -  `csebuetnlp/mT5_multilingual_XLSum`               
Toxicity Detection       -  `unitary/multilingual-toxic-xlm-roberta`          
Misinformation Detection -  `facebook/bart-large-mnli`                        
Transcription            -   OpenAI `whisper` (base model)                     
