import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import os
from faster_whisper import WhisperModel

# -------------------------------
# Load Faster-Whisper Base model
# -------------------------------
@st.cache_resource
def load_model():
    # Use faster-whisper base model
    model = WhisperModel(
        "base",
        device="cpu",         # M4 accelerates CPU/Metal internally
        compute_type="int8"   # fast + low RAM
    )
    return model

model = load_model()

st.title("🎤 Video Transcription (Faster-Whisper Base)")
st.write("Upload a video and get accurate transcription using the Whisper **base** model.")

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    st.video(uploaded)

    if st.button("Transcribe"):
        with st.spinner("Extracting audio..."):
            # Save the uploaded video to a temporary file
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded.read())
            temp_video.close()

            # Extract audio as WAV (16kHz mono)
            clip = VideoFileClip(temp_video.name)
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            clip.audio.write_audiofile(temp_audio.name, fps=16000, codec="pcm_s16le")
            clip.close()

        with st.spinner("Transcribing (Whisper base)..."):
            segments, info = model.transcribe(
                temp_audio.name,
                beam_size=1,            # fastest, good for base model
                word_timestamps=False    # change to True if needed
            )

        # Remove temp files
        os.remove(temp_video.name)
        os.remove(temp_audio.name)

        # --------------- Output ---------------
        st.subheader("📄 Transcription")
        final_text = ""

        for seg in segments:
            final_text += seg.text + " "

        st.write(final_text.strip())

        # Show detected language
        st.caption(f"Detected language: {info.language} (confidence {info.language_probability:.2f})")
