import cv2
import glob
import os
import librosa
from moviepy.editor import *

# Primary feature extraction from audio file
time_series, sample_rate = librosa.load("file_path")
seconds = librosa.get_duration(y=time_series, sr=sample_rate)


images_folder = 'file_path'
image_paths_ordered = []

# Compile .png files
for i in range(1, len(glob.glob(f'{images_folder}/*.png'))):
    filename = os.path.join(images_folder, f"img_{i}.png")
    image_paths_ordered.append(filename)
    
# Create video file
frames = len(image_paths_ordered)
fps = frames / seconds
frame = cv2.imread(image_paths_ordered[0])
height, width, layers = frame.shape

video = cv2.VideoWriter('video_output.avi', 
                        cv2.VideoWriter_fourcc(*'DIVX'), 
                        fps, 
                        (width, height))

for image in image_paths_ordered:
    print(image)
    print(cv2.imread(image).shape)
    video.write(cv2.imread(image))

video.release()
cv2.destroyAllWindows()

# Add music to video file
clip = VideoFileClip("video_output.avi")
audioclip = AudioFileClip("file_path")
videoclip = clip.set_audio(audioclip) 
videoclip.write_videofile("video_output_with_audio4.mp4")