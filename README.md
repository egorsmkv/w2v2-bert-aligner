# Align audio and text using wav2vec2-bert models

## Community

- **Discord**: https://bit.ly/discord-uds
- Speech Recognition: https://t.me/speech_recognition_uk
- Speech Synthesis: https://t.me/speech_synthesis_uk

## Install

```bash
uv venv --python 3.12

source .venv/bin/activate

uv pip install -r requirements.txt

# in development mode
uv pip install -r requirements-dev.txt

# check/format the code
ruff check --select I --fix
ruff format
```

## Test

### Download & convert the data

```bash
wget "https://github.com/egorsmkv/cv10-uk-testset-clean/releases/download/v1.0/filtered-cv10-test.zip"

unzip filtered-cv10-test.zip

wget https://raw.githubusercontent.com/egorsmkv/cv10-uk-testset-clean/refs/heads/main/rows.csv
```

### Run aligment

```bash
python alignment.py
```
