import os

# Importing the API must not require CUDA or the Chatterbox weights.
os.environ["TTS_SKIP_MODEL"] = "1"
