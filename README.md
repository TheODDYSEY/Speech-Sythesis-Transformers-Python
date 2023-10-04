# Speech Generation with SpeechT5 and HiFIGAN Vocoder

This Python script demonstrates how to use the SpeechT5 model and HiFIGAN vocoder to generate speech from text input. It includes the option to specify a particular speaker or use a random voice for speech synthesis.

## Prerequisites

- Python 3.x
- [Transformers](https://github.com/huggingface/transformers) library
- [Datasets](https://github.com/huggingface/datasets) library
- PyTorch
- SoundFile (soundfile) library

You can install the required Python libraries using `pip`:

```bash
pip install transformers datasets torch soundfile
```

## Usage

1. Save the script as a Python (.py) file, e.g., `speech_generation.py`.

2. Open a terminal or command prompt and navigate to the directory where the script is located.

3. Run the script using Python:

```bash
python speech_generation.py
```

The script will generate speech based on the text input and save the speech as an MP3 file.

## Functionality

- The script loads a pretrained SpeechT5 model and HiFIGAN vocoder using the Transformers library.

- It loads a dataset of speaker embeddings to associate specific speakers with the generated speech.

- You can specify a particular speaker for the generated speech by providing the speaker's name. The available speakers are defined in the `speakers` dictionary within the script.

- If you do not specify a speaker, the script will generate speech using a random voice.

- The generated speech is saved as an MP3 file with a filename that includes the speaker's name (if specified) or a random string, along with the first few words of the input text.

- The script uses a 16KHz sampling rate for the saved speech.

## Example Usage

1. Generate speech with a specific speaker (e.g., US female):

```python
save_text_to_speech("Python is my favorite programming language", speaker=speakers["slt"])
```

2. Generate speech with a random voice:

```python
save_text_to_speech("Python is my favorite programming language")
```

3. Generate speech with a challenging text for all available speakers:

```python
text = """In his miracle year, he published four groundbreaking papers. 
These outlined the theory of the photoelectric effect, explained Brownian motion, 
introduced special relativity, and demonstrated mass-energy equivalence."""
for speaker_name, speaker in speakers.items():
    output_filename = save_text_to_speech(text, speaker)
    print(f"Saved {output_filename}")
```

4. Generate speech with a random voice for challenging text:

```python
output_filename = save_text_to_speech(text)
print(f"Saved {output_filename}")
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
