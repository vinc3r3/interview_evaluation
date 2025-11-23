import streamlit as st
import plotly.graph_objects as go
import os
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from faster_whisper import WhisperModel
from openai import OpenAI
import json
from typing import Dict, Any

model = WhisperModel(
    "base",                # Model type (e.g., base, small, medium, large)
    device="cpu",          # Device to run the model (e.g., "cpu" or "cuda")
    compute_type="int8"    # Optimization for performance
)

# Initialize OpenAI client (will be updated with user's API key)
client = None

def get_personality_traits_gpt(text: str, model: str = "gpt-5-nano-2025-08-07") -> Dict[str, Any]:
    """
    Calls OpenAI API to analyze personality traits from text using the Big Five model.
    
    Args:
        text (str): The text to analyze
        model (str): OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    
    Returns:
        Dict containing traits and scores, or error information
    """
    
    prompt = f"""You are a careful, evidence-based psychologist who specialises in the Big Five (OCEAN) personality model.  
Your job is to infer approximate **Big Five trait scores** from a piece of text.

---

### 1. Task

Given an input **TEXT** that reflects a person's writing (messages, essays, posts, etc.), estimate their stable personality tendencies along the **Big Five** dimensions:

- Agreeableness  
- Neuroticism  
- Openness to Experience  
- Conscientiousness  
- Extraversion  

You must output **only** numeric scores between **0.0 and 1.0** (inclusive), where:

- 0.0 = extremely low on this trait  
- 0.5 = average / unsure  
- 1.0 = extremely high on this trait  

Use two or three decimal places.

---

### 2. Conceptual guides (use these when interpreting the text)

**Openness to Experience**  
- High: appreciates art, emotion, beauty, imagination, curiosity, variety, and unusual ideas; likes trying new things; creative, intellectually curious, uses rich/vivid language, reflects on abstract ideas; may hold unconventional beliefs and seek intense or euphoric experiences.  
- Low: prefers routine and familiarity; pragmatic, data-driven, and focused on practicality; disinterested in abstract or imaginative topics; can appear dogmatic or closed-minded.

**Conscientiousness**  
- High: self-disciplined, organised, dutiful, goal- and achievement-oriented; likes order, schedules, and planning; completes tasks promptly, pays attention to details, takes obligations seriously; behaviour is controlled and reliable.  
- Low: flexible and spontaneous but can be disorganised, messy, unreliable; procrastinates, forgets or abandons tasks; tends to "wing it" instead of planning carefully.

**Extraversion**  
- High: energetic, talkative, outgoing; enjoys social interaction and being around people; seeks external stimulation; starts conversations, likes being the centre of attention, active and enthusiastic in groups.  
- Low (introversion): quiet, reserved, low-key; prefers depth over breadth in social contacts; may avoid being centre of attention, keeps in the background; needs more time alone and less external stimulation, but is not necessarily unfriendly or depressed.

**Agreeableness**  
- High: kind, considerate, trusting and trustworthy, generous, compassionate; interested in others, takes time to help, feels others' emotions, makes people feel at ease; values social harmony and cooperation, optimistic about others' motives.  
- Low: puts own interests first; more skeptical or suspicious of others' motives; can be unfriendly, blunt, competitive, argumentative, or uncooperative; less concerned with others' problems or well-being.

**Neuroticism**  
- High: emotionally volatile and reactive; prone to strong negative emotions (anxiety, worry, anger, sadness); easily stressed or upset; mood swings, frequent irritability, pessimism; interprets situations as threatening or overwhelming, ruminates on problems.  
- Low (emotional stability): calm, even-tempered; less easily upset or stressed; negative emotions fade more quickly; generally emotionally stable and resilient (this does **not** automatically mean very positive or cheerful—that is more related to extraversion).

---

### 3. Important instructions

1. **Base your judgement only on the TEXT.**  
   - Do not assume traits that are not supported by evidence in the text.  
   - If the text is very short or ambiguous for a trait, keep that trait closer to **0.5** (uncertain/average).

2. **Focus on stable tendencies**, not temporary moods.  
   - Look for patterns in how the person talks about themselves, others, work, feelings, plans, and experiences.

3. **Use the full 0–1 range** when justified.  
   - Very strong, repeated signals of a trait → move closer to 0.1 or 0.9+.  
   - Neutral or mixed signals → keep near 0.4–0.6.  
   - Strong evidence of the opposite pole → move toward 0.0–0.2.

4. **No explanation in the final answer.**  
   - Internally you may reason, but your final output must strictly follow the required JSON format below, with no extra text.

---

### 4. Output format (MUST follow exactly)

Return **exactly one JSON object** with:

- `"traits"`: array of trait names in this exact order  
  `["Agreeableness", "Neuroticism", "Openness", "Conscientiousness", "Extraversion"]`
- `"scores"`: array of 5 floating-point numbers in the same order, each between 0.0 and 1.0 (inclusive), with 2–3 decimal places.

**Example of valid output format (structure only):**

    {{{{
      "traits": ["Agreeableness", "Neuroticism", "Openness", "Conscientiousness", "Extraversion"],
      "scores": [0.72, 0.31, 0.84, 0.59, 0.46]
    }}}}

Do **not** add comments, explanations, or any additional keys.

---

### 5. Now analyse this text

TEXT:
{text}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional psychologist specializing in personality assessment using the Big Five model."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response for debugging
            return {
                "error": "JSON parsing failed",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__
        }

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

# OpenAI API Key Input
api_key_input = st.text_input(
    "Enter your OpenAI API Key:",
    type="password",
    placeholder="sk-proj-...",
    help="Your API key will be used to analyze personality traits from the transcription."
)

# Update the OpenAI client with the user-provided API key
if api_key_input:
    try:
        client = OpenAI(api_key=api_key_input)
        # Test the API key with a simple request
        test_response = client.models.list()
        st.success("✅ Your API key is valid!")
    except Exception as e:
        st.error(f"❌ Invalid API key: {str(e)}")
        client = None

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Middle column - narrower
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
        # Save the video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_to_process.read())
        temp_video.close()

        # ===== TRANSCRIPTION =====
        with st.spinner("Extracting audio..."):
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

        # Remove audio temp file
        os.remove(temp_audio.name)

        # --------------- Transcription Output ---------------
        st.subheader("📄 Transcription")
        transcription = ""

        for seg in segments:
            transcription += seg.text + " "

        st.write(transcription.strip())

        st.markdown("---")

        # ===== PERSONALITY TRAIT ANALYSIS =====
        with st.spinner("Analyzing personality traits..."):
            personality_result = get_personality_traits_gpt(transcription.strip())
        
        if "error" not in personality_result:
            traits = personality_result.get("traits", ["Agreeableness", "Neuroticism", "Openness", "Conscientiousness", "Extraversion"])
            scores = personality_result.get("scores", [0.5, 0.5, 0.5, 0.5, 0.5])
            
            # Create a dataframe with personality traits and scores
            import pandas as pd
            df_personality = pd.DataFrame({
                "Personality Trait": traits,
                "Score": scores
            })
            
            st.subheader("🗣️ Big 5 Personality Trait Table")
            st.dataframe(df_personality)
            
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

            st.subheader("🗣️ Big 5 Personality Trait Scores")
            st.plotly_chart(fig_big5, use_container_width=True)
        else:
            st.error(f"Error analyzing personality traits: {personality_result.get('error', 'Unknown error')}")

        st.markdown("---")

        # ===== EMOTION ANALYSIS =====
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

                st.markdown("---")

                # ===== BEHAVIORAL ASSESSMENT =====
                with st.spinner("Generating behavioral assessment..."):
                    # Convert dataframes to string format for the prompt
                    emotion_df_str = df_aggregated.to_string(index=False)
                    personality_df_str = df_personality.to_string(index=False)
                    
                    behavioral_prompt = f"""You are a senior organizational psychologist and expert interviewer specializing in competency-based hiring. Your job is to perform a BEHAVIOURAL ASSESSMENT of a candidate based on and give a easy to understand report:

1) The transcription of their interview answer (verbal content and wording),
2) Their Big Five personality profile inferred from text,
3) Their emotional probabilities over time inferred from video.

You are given three inputs:

transcription="{transcription.strip()}"

df_emotions=
"{emotion_df_str}"

df_personality=
"{personality_df_str}"

Where:
- `transcription` is the candidate's spoken answer as plain text.
- `df_emotions` is a table with columns:
    - timestamp (in seconds),
    - angry, disgust, fear, happy, sad, surprise, neutral  
  Each emotion column contains a probability from 0 to 1 at a given timestamp.
- `df_personality` is a table with two columns:
    - Personality Trait (Agreeableness, Neuroticism, Openness, Conscientiousness, Extraversion),
    - Score (a continuous value from 0 to 1, where 0 = very low and 1 = very high).

--------------------
YOUR TASK
--------------------
1. Interpret the transcription:
   - Identify the candidate's self-presentation style (confident, modest, exaggerated, vague, specific, etc.).
   - Highlight key behavioural signals in the text: diligence, initiative, teamwork, reliability, learning attitude, problem solving, communication clarity, etc.
   - Comment on how authentic and consistent their claims sound.

2. Interpret the Big Five personality profile:
   - Explain what each provided score suggests about the candidate's behaviour at work.
   - Focus on:
     - Conscientiousness: reliability, planning, attention to detail.
     - Agreeableness: teamwork, conflict management, empathy.
     - Neuroticism: emotional stability, reaction to stress.
     - Extraversion: communication, energy in social contexts.
     - Openness: learning, creativity, adaptability.
   - Relate these traits explicitly to potential job performance (no generic textbook definitions; always tie back to work behaviour).

3. Interpret the emotional timeline from `df_emotions`:
   - Look at how emotions change over time and identify patterns:
     - Are positive emotions (e.g., happy) dominant or not?
     - Are there spikes in negative emotions (angry, fear, sad) at particular points?
     - Is there a strong neutral baseline or a lot of fluctuation?
   - Describe what this might mean behaviourally during an interview:
     - Confidence vs. anxiety,
     - Authentic enthusiasm vs. forced positivity,
     - Composure vs. tension or defensiveness.
   - DO NOT list raw numbers or reproduce the table. Summarize patterns in plain language.

4. Integrate everything into a behavioural assessment in a simple and clear text:
   - Combine verbal content, personality traits, and emotional patterns.
   - Evaluate:
     - Reliability and work ethic,
     - Teamwork and communication,
     - Stress tolerance and emotional control,
     - Learning and adaptability,
     - Overall job fit from a behavioural standpoint.
   - If there are contradictions (e.g., very confident words but nervous emotional signals), point them out and explain what they might imply.
   - Use **bold** text to highlight section headings and important information in your response.
   - Make section headings bigger than other text, clear and distinct.
   - Provide output with no more than 300 words.

5. Give a final hiring recommendation SCORE OUT OF 100:
   - This is a "recommendation strength" score: 0 = strongly not recommended, 100 = extremely strong recommendation.
   - Base the score on all three sources of evidence (speech, Big Five, emotions).
   - Be realistic and nuanced (avoid giving 0 or 100 except in extreme, obvious cases).
   - Briefly justify why you chose that score in terms of job-relevant behaviour.

--------------------
OUTPUT FORMAT
--------------------
Follow this exact structure in your response:

**Behavioural Assessment Summary**
Write 1 concise paragraph that:
- Summarize who this candidate appears to be as a professional,
- Link the transcription content with the Big Five profile,
- Highlight the most important behavioural strengths and possible risks.\n"
"\n"
"\n"
**Emotional and Personality Insights**
Write 1 paragraph that:
- Describe the main emotional patterns over time (e.g., "mostly calm with brief spikes of anxiety when discussing X"),
- Explain how these emotional patterns interact with the Big Five scores (e.g., high conscientiousness + visible tension, etc.),
- Comment on authenticity, composure, and how they might behave in real work situations.\n"
"\n"
"\n"
**Recommendation Score**
On a separate line, output:
✔️ <NUMBER> / 100 — <one-sentence recommendation>

Where:
- <NUMBER> is your recommendation score out of 100 (integer or one decimal place),
- The sentence clearly states whether you recommend inviting the candidate further and for what type of role they seem best suited.

Do NOT repeat the raw tables. Do NOT say "as an AI model." Speak as a human expert interviewer giving a professional behavioural assessment. Never use em dash in your output"""

                    try:
                        behavioral_response = client.chat.completions.create(
                            model="gpt-5.1-2025-11-13",
                            temperature=0.5,
                            max_completion_tokens=600,
                            messages=[
                                {"role": "system", "content": "You are a senior organizational psychologist and expert interviewer specializing in competency-based hiring."},
                                {"role": "user", "content": behavioral_prompt}
                            ]
                        )
                        
                        behavioral_assessment = behavioral_response.choices[0].message.content.strip()
                        
                        st.subheader("🧠 Behavioral Assessment and Explainability")
                        
                        # Apply custom CSS for better readability (default spacing)
                        st.markdown("""
                            <style>
                            .behavioral-assessment {
                                font-size: 1.1rem;
                                padding: 20px;
                                background-color: rgba(255, 255, 255, 0.05);
                                border-radius: 10px;
                                margin: 20px 0;
                            }
                            .behavioral-assessment h3 {
                                font-size: 1.3rem;
                                color: #4da6ff;
                                font-weight: 600;
                            }
                            .behavioral-assessment strong {
                                font-weight: 600;
                                color: #ffffff;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        # Convert markdown bold to HTML for proper styling
                        import re
                        
                        # First, convert **text** at the start of a line (section headers) to <h3>
                        # This pattern matches ** at the start of a line or after newline
                        behavioral_assessment_html = re.sub(r'(^|\n)\*\*([^*]+)\*\*', r'\1<h3>\2</h3>', behavioral_assessment)
                        
                        # Then convert remaining **text** (inline bold) to <strong> tags
                        behavioral_assessment_html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', behavioral_assessment_html)
                        
                        # Display the assessment with custom styling
                        st.markdown(f'<div class="behavioral-assessment">{behavioral_assessment_html}</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating behavioral assessment: {str(e)}")

        else:
            st.error("No emotions detected in the video.")

        # Remove the temporary video file
        os.remove(temp_video.name)

    else:
        st.error("Please upload a video or ensure the default video exists.")

st.markdown("---")

# Note: Big 5 Personality Trait Analysis chart is now generated dynamically
# when the "Analyse the Interview" button is pressed
