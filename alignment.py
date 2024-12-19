from os.path import basename, exists
from typing import List

import ctc_segmentation
import numpy as np
import sphn
import torch
from natsort import natsorted
from tqdm import tqdm
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer

model_name = "Yehor/w2v-bert-uk-v2.1"
SAMPLERATE = 16_000

processor = Wav2Vec2BertProcessor.from_pretrained(model_name)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2BertForCTC.from_pretrained(model_name)
model = model.to(device="cuda")


CSV_FILE = "/home/yehor/projects/github/w2v2-bert-aligner/rows.csv"
WAV_DIR = "/home/yehor/projects/github/w2v2-bert-aligner/filtered-cv10-test"
ALIGNED_DATA_DIR = "/home/yehor/projects/github/w2v2-bert-aligner/aligned-data"


def align_with_transcript(
    audio: np.ndarray,
    transcripts: List[str],
    samplerate: int = SAMPLERATE,
    model: Wav2Vec2BertForCTC = model,
    processor: Wav2Vec2BertProcessor = processor,
    tokenizer: Wav2Vec2CTCTokenizer = tokenizer,
):
    assert audio.ndim == 1

    # Run prediction, get logits and probabilities
    inputs = processor(
        audio, sampling_rate=16_000, return_tensors="pt", padding="longest"
    )
    inputs = inputs.to(device="cuda")

    with torch.inference_mode():
        logits = model(inputs.input_features).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)

    # Tokenize transcripts
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = tokenizer(transcript.replace("\n", " ").lower())["input_ids"]
        tok_ids = np.array(tok_ids, dtype=int)
        tokens.append(tok_ids[tok_ids != unk_id])

    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(
        config, tokens
    )
    timings, char_probs, _ = ctc_segmentation.ctc_segmentation(
        config, probs.numpy(), ground_truth_mat
    )
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, transcripts
    )
    return [
        {"text": t, "start": p[0], "end": p[1], "conf": p[2]}
        for t, p in zip(transcripts, segments)
    ]


def get_word_timestamps(
    audio: np.ndarray,
    samplerate: int = SAMPLERATE,
    model: Wav2Vec2BertForCTC = model,
    processor: Wav2Vec2BertProcessor = processor,
    tokenizer: Wav2Vec2CTCTokenizer = tokenizer,
):
    assert audio.ndim == 1

    # Run prediction, get logits and probabilities
    inputs = processor(
        audio, sampling_rate=16_000, return_tensors="pt", padding="longest"
    )
    with torch.inference_mode():
        logits = model(inputs.input_features).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)

    predicted_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.decode(predicted_ids)

    # Split the transcription into words
    words = pred_transcript.split(" ")

    # Align
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio.shape[0] / probs.size()[0] / samplerate

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
        config, probs.numpy(), ground_truth_mat
    )
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, words
    )
    return [
        {"text": w, "start": p[0], "end": p[1], "conf": p[2]}
        for w, p in zip(words, segments)
    ]


with open(CSV_FILE) as f:
    for line in tqdm(natsorted(f)):
        path, transcript = line.strip().split(",")
        if path == "path":
            continue

        filename = basename(path)
        file_id = filename.split(".")[0]
        filename = f"{WAV_DIR}/{file_id}.wav"
        aligned_file = f"{ALIGNED_DATA_DIR}/{file_id}.wav"

        if exists(aligned_file):
            continue

        audio_data, _ = sphn.read(filename)
        audio = audio_data[0]

        transcripts = [
            transcript.strip(),
        ]

        results = align_with_transcript(audio, transcripts)
        result = results[0]

        start_sec = result["start"]
        end_sec = result["end"]

        reader = sphn.FileReader(filename)
        data = reader.decode(start_sec, end_sec - start_sec)
        sphn.write_wav(aligned_file, data, reader.sample_rate)
