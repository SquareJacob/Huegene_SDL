import subprocess
import os
from moviepy.editor import *

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the executable
executable_path = os.path.join(current_dir, 'Huegene.exe')

# Run the executable
process = subprocess.Popen([executable_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Read the output in real-time
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

# Get any remaining output
stderr_output = process.stderr.read()
if stderr_output:
    print("STDERR:", stderr_output.strip())
img = os.listdir("Images")
imgPath = ["Images/Image" + str(i) + ".bmp" for i in range(len(img))]
clips = [ImageClip(m).set_duration(0.03)
      for m in imgPath]
concat_clip = concatenate_videoclips(clips)
concat_clip.write_videofile("test.mp4", fps=24)
for m in imgPath:
    os.remove(m)

