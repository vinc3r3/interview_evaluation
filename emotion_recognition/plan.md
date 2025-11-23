Emotion recognition entegration:

1. Make sure that when a new button "analyze emotion" is pressed, then a video uploaded to streamlit or default video (if nothing is uploaded) gets processed by video_inference.py. Then the datafreame df from function process_video in video_inference.py is sent to a variable in main.py and printed as a dataframe in the streamlit interface
2. Take the necessary columns of that datafreame and visualize a new emotion graph and test on all 3 videos
3. Delete everything unnecessary and leave only functional things