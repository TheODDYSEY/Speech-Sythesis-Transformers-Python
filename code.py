from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import random
import string
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
# load the processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# load the model
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
# load the vocoder, that is the voice encoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
# we load this dataset to get the speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker ids from the embeddings dataset
speakers = {
    'awb': 0,     # Scottish male
    'bdl': 1138,  # US male
    'clb': 2271,  # US female
    'jmk': 3403,  # Canadian male
    'ksp': 4535,  # Indian male
    'rms': 5667,  # US male
    'slt': 6799   # US female
}