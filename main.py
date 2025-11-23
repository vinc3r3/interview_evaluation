import streamlit as st
import plotly.graph_objects as go
import os
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel

model = WhisperModel(
    "base",                # Model type (e.g., base, small, medium, large)
    device="cpu",          # Device to run the model (e.g., "cpu" or "cuda")
    compute_type="int8"    # Optimization for performance
)

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Interview Emotion & Personality Analysis",
    layout="wide",
)



# Center the title using Streamlit's markdown with HTML
st.markdown(
    "<h1 style='text-align: center;'>AI-powered Interview Evaluation System</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Upload or use the demo video:")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)
    else:
        # Default video set to a local file
        default_video_path = "/Users/nursultanatymtay/Desktop/Senior Project/Project/interview_evaluation/interview_example.mp4"

        if os.path.exists(default_video_path):
            st.video(default_video_path)
        else:
            st.error("Default video file not found. Please upload a video.")



# Move the analyse button below the video (full width, directly under the video area)
if st.button("Analyse the Interview"):
    video_to_process = uploaded_video

    if uploaded_video is None:
        # Use default video if no video is uploaded
        default_video_path = "/Users/nursultanatymtay/Desktop/Senior Project/Project/interview_evaluation/interview_example.mp4"
        if os.path.exists(default_video_path):
            video_to_process = open(default_video_path, "rb")
        else:
            st.error("Default video file not found. Please upload a video.")

    if video_to_process is not None:
        with st.spinner("Extracting audio..."):
            # Save the video to a temporary file
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(video_to_process.read())
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

    else:
        st.error("Please upload a video or ensure the default video exists.")

st.markdown("---")

# ---------------- Emotion Analysis from Video ----------------

# Same data as in your HTML example
emotion_spectrum_data = {
    "Happiness": [0.2, 0.3, 0.7, 0.8, 0.6, 0.7, 0.9],
    "Sadness":   [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.1],
    "Anger":     [0.05, 0.1, 0.05, 0.1, 0.15, 0.1, 0.05],
    "Surprise":  [0.3, 0.4, 0.2, 0.5, 0.3, 0.6, 0.4],
    "Disgust":   [0.02, 0.05, 0.03, 0.08, 0.05, 0.06, 0.04],
    "Fear":      [0.01, 0.03, 0.02, 0.04, 0.03, 0.02, 0.01],
}
time_labels = ["0s", "10s", "20s", "30s", "40s", "50s", "60s"]

fig_emotion = go.Figure()

# Merge all emotions into a single loop with specific colors
emotion_colors = {
    "Happiness": "#FFFF00",  # Vibrant Yellow
    "Sadness": "#0000FF",   # Blue
    "Anger": "#FF0000",     # Bright Red
    "Surprise": "#00FFFF",  # Light Cyan (bright hue)
    "Disgust": "#008000",   # Classic Green
    "Fear": "#BB00FF"       # Purple
}

# Merge all emotions into a single loop
for emo, data in emotion_spectrum_data.items():
    fig_emotion.add_trace(
        go.Scatter(
            x=time_labels,
            y=data,
            mode="lines+markers",
            name=emo,
            line=dict(width=2, shape="spline", color=emotion_colors.get(emo, "#000000")),  # Default to black if not found
            marker=dict(size=6),
        )
    )

fig_emotion.update_layout(
    xaxis_title="Time",
    yaxis_title="Emotion Probability",
    yaxis=dict(range=[0, 1]),
    legend=dict(orientation="h", y=-0.2),
    height=400,
)

# ---------------- Big 5 Personality Trait Analysis from Text ----------------

traits = ["Agreeableness", "Neuroticism", "Openness", "Conscientiousness", "Extraversion"]
scores = [0.7, 0.3, 0.8, 0.6, 0.5]

# Close the loop for radar plot
traits_closed = traits + [traits[0]]
scores_closed = scores + [scores[0]]

fig_big5 = go.Figure()

fig_big5.add_trace(
    go.Scatterpolar(
        r=scores_closed,
        theta=traits_closed,
        fill="toself",
        name="Big 5 Profile",
        fillcolor="rgba(0, 123, 255, 0.2)",  # Subtle blue fill
        line=dict(color="rgba(0, 123, 255, 1)", width=2),  # Blue border
    )
)

fig_big5.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            dtick=0.2,
            gridcolor="rgba(255, 255, 255, 0.1)",  # Subtle grid lines
            linecolor="rgba(255, 255, 255, 0.3)",  # Subtle axis lines
        ),
        bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
    ),
    showlegend=False,
    height=450,
    paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent paper background
    plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent plot background
)

# Create columns for side-by-side charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎨 Emotion Analysis")
    st.plotly_chart(fig_emotion, use_container_width=True, height=300)  # Reduced height

with col2:
    st.subheader("🗣️ Big 5 Personality Trait Analysis")
    st.plotly_chart(fig_big5, use_container_width=True, height=300)  # Reduced height

# ---------------- Emotion Analysis from Video (Button and Processing) ----------------

if st.button("Analyze Emotion"):
    video_to_process = uploaded_video

    if uploaded_video is None:
        # Use default video if no video is uploaded
        default_video_path = "/Users/nursultanatymtay/Desktop/Senior Project/Project/interview_evaluation/interview_example.mp4"
        if os.path.exists(default_video_path):
            video_to_process = open(default_video_path, "rb")
        else:
            st.error("Default video file not found. Please upload a video.")

    if video_to_process is not None:
        # Save the video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_to_process.read())
        temp_video.close()

        # Call the video_inference.py script
        import subprocess
        import pandas as pd
        from emotion_recognition.video_inference import VideoEmotionDetector

        model_path = "/Users/nursultanatymtay/Desktop/Senior Project/Project/interview_evaluation/emotion_recognition/best_model_fixed.pth"
        detector = VideoEmotionDetector(model_path, device="cpu")

        with st.spinner("Processing video for emotion analysis..."):
            df = detector.process_video(temp_video.name, sample_rate=10)

        # Display the dataframe in Streamlit
        if not df.empty:
            # Process the dataframe to average every 15 rows (5 seconds group)
            df = df.drop(columns=['dominant_emotion', 'confidence', 'frame_number'], errors='ignore')
            df['group'] = df['timestamp'] // 5  # Group rows by intervals of 5 seconds
            df_aggregated = df.groupby('group').agg({
                'timestamp': 'first',  # Keep the first timestamp of the group
                **{col: 'mean' for col in df.columns if col not in ['timestamp', 'group']}
            }).reset_index(drop=True)

            # Update the timestamp column to represent the start of each group
            df_aggregated['timestamp'] = df_aggregated.index * 5

            # Drop the group index and reset the dataframe
            df_aggregated = df_aggregated.reset_index(drop=True)

            st.subheader("🎨 Emotion Analysis Table")
            st.dataframe(df_aggregated)


            # Update the emotion graph with new data from the aggregated dataframe
            if not df_aggregated.empty:
                # Update the emotion graph with new data from the aggregated dataframe
                emotion_spectrum_data = {
                    "Happiness": df_aggregated["happy"].tolist(),
                    "Sadness": df_aggregated["sad"].tolist(),
                    "Anger": df_aggregated["angry"].tolist(),
                    "Surprise": df_aggregated["surprise"].tolist(),
                    "Disgust": df_aggregated["disgust"].tolist(),
                    "Fear": df_aggregated["fear"].tolist(),
                    "Neutral": df_aggregated["neutral"].tolist(),  # Add neutral emotion
                }
                time_labels = [f"{int(ts)}s" for ts in df_aggregated["timestamp"]]

                emotion_colors = {
                    "Happiness": "#FFFF00",  # Vibrant Yellow
                    "Sadness": "#0000FF",   # Blue
                    "Anger": "#FF0000",     # Bright Red
                    "Surprise": "#00FFFF",  # Light Cyan (bright hue)
                    "Disgust": "#008000",   # Classic Green
                    "Fear": "#BB00FF",       # Purple
                    "Neutral": "#808080"    # Gray for neutral
                }

                fig_emotion = go.Figure()

                for emo, data in emotion_spectrum_data.items():
                    fig_emotion.add_trace(
                        go.Scatter(
                            x=time_labels,
                            y=data,
                            mode="lines+markers",
                            name=emo,
                            line=dict(width=2, shape="spline", color=emotion_colors.get(emo, "#000000")),  # Default to black if not found
                            marker=dict(size=6),
                        )
                    )

                fig_emotion.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Emotion Probability",
                    yaxis=dict(range=[0, 1]),
                    legend=dict(orientation="h", y=-0.2),
                    height=400,
                )

                st.subheader("🎨 Emotions Probabilities through Time")
                st.plotly_chart(fig_emotion, use_container_width=True)

        else:
            st.error("No emotions detected in the video.")

        # Remove the temporary video file
        os.remove(temp_video.name)
