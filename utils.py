import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
import evaluate
import inflect
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio, load_dataset
from torchaudio.transforms import SpeedPerturbation
from jiwer import Compose, RemovePunctuation, ToLowerCase, RemoveMultipleSpaces

import params

metric = evaluate.load("wer")
inflect_engine = inflect.engine()

# Custom transformation to convert numbers to words
class NumbersToWords:
    def __call__(self, text):
        if isinstance(text, list):
            words = []
            for item in text :
                words.extend(item.split())
        else :
            words = text.split()
        converted = [inflect_engine.number_to_words(word).replace("-", "") if word.isdigit() else word for word in words]
        return converted

# Define the transformation pipeline
custom_transform = Compose([
    RemoveMultipleSpaces(),
    RemovePunctuation(),
    ToLowerCase(),
    NumbersToWords()
])

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt").to(params.torch_dtype)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def prepare_dataset(batch, feature_extractor, augmentor=None):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    if augmentor is not None :
        # Convert the NumPy array to a PyTorch tensor (shares memory)
        audio_array_float32 = audio["array"].astype(np.float32)
        audio_tensor = torch.from_numpy(audio_array_float32)
        # Apply augmentation
        augmented_audio_tensor = augmentor(audio_tensor)
        # Convert the augmented tensor back to a NumPy array (shares memory)
        augmented_audio_array = augmented_audio_tensor[0].numpy()
        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(augmented_audio_array, sampling_rate=audio["sampling_rate"]).input_features[0]
    else :
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    return batch

def prepare_dataset_from_disk(batch, feature_extractor, tokenizer, augmentor=None) :
    batch = prepare_dataset(batch, feature_extractor, augmentor)
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def filter_Torgo_dataset(element) :
    file_name = element['audio']['path']
    if "headMic" in file_name:
        return True
    else :
        return False

def prepare_Torgo_dataset(batch, feature_extractor, tokenizer=None, augmentor=None):
    batch = prepare_dataset(batch, feature_extractor, augmentor)
    uid = batch["audio"]['path'].split("_")[0]
    if 'C' in uid:
        batch['severity'] = 0
    else:
        batch['severity'] = params.TORGO_dys_speaker_dict[uid]
    # encode target text to label ids
    batch['uid'] = uid
    batch['sentence'] = batch['transcription']
    if tokenizer is not None :
        batch["labels"] = tokenizer(batch['sentence']).input_ids
    return batch

def load_dataset_for_ASR_without_prepare(dataset_string, speakers, dataset_dir=Path(), forced_recreate=False) :
    if dataset_string == params.UASPEECH :
        file_paths, texts, labels = get_UASpeech_as_list(params.speakers_dict[speakers], params.uaspeech_dir)
    elif dataset_string == params.TORGO:
        dataset_dir_final = params.torgo_dir if dataset_dir == Path() else dataset_dir
        file_paths, texts, labels = get_TrogoGenerated_as_list(dataset_dir_final)
    elif dataset_string == params.LACICON:
        dataset_dir_final = params.laci_control_dir if dataset_dir == Path() else dataset_dir
        file_paths, texts, labels = get_LaciControl_as_list(dataset_dir_final)
    elif dataset_string == params.LACIDYS:
        dataset_dir_final = params.laci_dys_dir if dataset_dir == Path() else dataset_dir
        file_paths, texts, labels = get_LaciDys_as_list(dataset_dir_final)
    elif dataset_string == params.SZEGEDYS:
        file_paths, texts, labels = get_SzegedDys_as_list(dataset_dir)
    elif dataset_string == params.HUNDYS:
        file_paths, texts, labels = get_HunDys_as_list(dataset_dir)
    dataset = Dataset.from_dict({"audio": file_paths, "severity": labels, "sentence": texts}).cast_column("audio",
        Audio(sampling_rate=params.SAMPLING_RATE))
    return dataset

def load_dataset_for_ASR(dataset_string, speakers, dataset_dir=Path(), fn_kwargs=None, forced_recreate=False) :
    dataset = load_dataset_for_ASR_without_prepare(dataset_string, speakers, dataset_dir, forced_recreate)
    if dataset_string is params.TORGO and (not os.path.exists(params.processed_paths[speakers]) or forced_recreate) :
        dataset = dataset.map(prepare_Torgo_dataset, fn_kwargs=fn_kwargs, remove_columns=["audio"])
    else :
        dataset = dataset.map(prepare_dataset_from_disk, fn_kwargs=fn_kwargs, remove_columns=["audio"])
    return dataset

def get_LaciControl_as_list(data_dir) :
    texts = []
    labels = []
    file_paths = []
    for path_Laci_file in os.listdir(data_dir+"/text/") :
        basename = path_Laci_file.split(".")[0]
        wavname = data_dir+"/wav/"+basename+"_volnorm_cut_ultrasound.wav"
        if not os.path.isfile(wavname) :
            print(wavname)
        else :
            file_paths.append(wavname)
            textname = data_dir + "/text/" + path_Laci_file
            try :
                with open(textname, 'r', encoding='utf-8') as file:
                    text = str(file.read().rstrip('\n').capitalize())
            except UnicodeDecodeError :
                with open(textname, 'r', encoding='latin-1') as file:
                    text = str(file.read().rstrip('\n').capitalize())
                char_fixes = {
                    'û': 'ű', 'õ': 'ő',
                    'ã': 'á', 'â': 'á', 'ê': 'é', 'î': 'í', 'ô': 'ó',
                    'Û': 'Ű', 'Õ': 'Ő',
                    'Ã': 'Á', 'Â': 'Á', 'Ê': 'É', 'Î': 'Í', 'Ô': 'Ó',
                }
                for wrong_char, correct_char in char_fixes.items():
                    text = text.replace(wrong_char, correct_char)
            texts.append(text)
            labels.append(0)
    return file_paths, texts, labels

def get_Bea_as_list(data_dir) :
    texts = []
    labels = []
    file_paths = []
    for wavpath in Path(data_dir).iterdir() :
        if not wavpath.is_file() :
            print(wavpath)
        else :
            file_paths.append(str(wavpath))
            texts.append(" ")
            labels.append(0)
    return file_paths, texts, labels

def get_Szindbad_as_list(data_dir) :
    texts = []
    labels = []
    file_paths = []
    for textpath in Path(data_dir, "text").iterdir() :
        wavname = textpath.stem.replace('.mfcc', '') + '.wav'
        wavpath = Path(data_dir) / "wav" / wavname
        if not wavpath.is_file() :
            print(wavname)
        else :
            file_paths.append(str(wavpath))
            try :
                text = textpath.read_bytes().decode('utf8').lower().replace('\r', '').replace('\n', ' ').strip()
            except UnicodeDecodeError :
                text = textpath.read_bytes().decode('latin-1').lower().replace('\r', '').replace('\n', ' ').strip()
                char_fixes = {
                    'û': 'ű', 'õ': 'ő',
                    'ã': 'á', 'â': 'á', 'ê': 'é', 'î': 'í', 'ô': 'ó',
                    'Û': 'Ű', 'Õ': 'Ő',
                    'Ã': 'Á', 'Â': 'Á', 'Ê': 'É', 'Î': 'Í', 'Ô': 'Ó',
                }
                for wrong_char, correct_char in char_fixes.items():
                    text = text.replace(wrong_char, correct_char)
            texts.append(text)
            labels.append(0)
    return file_paths, texts, labels

def get_LaciDys_as_list(data_dir) :
    text_directory_path = Path(data_dir) / "text"
    wav_directory_path = Path(data_dir) / "wav"
    texts = []
    labels = []
    file_paths = []
    for path_Laci_file_text in text_directory_path.iterdir() :
        basename = "_".join(path_Laci_file_text.stem.split("_")[1:-1])
        for path_Laci_file_wav in wav_directory_path.iterdir() :
            if basename in path_Laci_file_wav.name :
                print(path_Laci_file_text, path_Laci_file_wav)
                file_paths.append(str(path_Laci_file_wav))
                try :
                    with open(path_Laci_file_text, 'r', encoding='utf-8') as file:
                        text = str(file.read().rstrip('\n').capitalize())
                except UnicodeDecodeError :
                    with open(path_Laci_file_text, 'r', encoding='latin-1') as file:
                        text = str(file.read().rstrip('\n').capitalize())
                    char_fixes = {
                        'û': 'ű', 'õ': 'ő',
                        'ã': 'á', 'â': 'á', 'ê': 'é', 'î': 'í', 'ô': 'ó',
                        'Û': 'Ű', 'Õ': 'Ő',
                        'Ã': 'Á', 'Â': 'Á', 'Ê': 'É', 'Î': 'Í', 'Ô': 'Ó',
                    }
                    for wrong_char, correct_char in char_fixes.items():
                        text = text.replace(wrong_char, correct_char)
                texts.append(text)
                labels.append(3)
    return file_paths, texts, labels

def get_HunDys_as_list(wav_directory_path=Path(params.hundys_dir)) :
    if isinstance(wav_directory_path, str):
        wav_directory_path = Path(wav_directory_path)
    text_directory_path = wav_directory_path.parent / "text"
    texts = []
    labels = []
    file_paths = []
    for path_file_wav in wav_directory_path.iterdir():
        label = int(path_file_wav.stem.split("_")[1])-1
        text_filename = path_file_wav.stem+".txt"
        path_file_text = text_directory_path / text_filename
        file_paths.append(str(path_file_wav))
        try :
            with open(path_file_text, 'r', encoding='utf-8') as file:
                text = str(file.read().rstrip('\n').capitalize())
        except UnicodeDecodeError :
            with open(path_file_text, 'r', encoding='latin-1') as file:
                text = str(file.read().rstrip('\n').capitalize())
            char_fixes = {
                'û': 'ű', 'õ': 'ő',
                'ã': 'á', 'â': 'á', 'ê': 'é', 'î': 'í', 'ô': 'ó',
                'Û': 'Ű', 'Õ': 'Ő',
                'Ã': 'Á', 'Â': 'Á', 'Ê': 'É', 'Î': 'Í', 'Ô': 'Ó',
            }
            for wrong_char, correct_char in char_fixes.items():
                text = text.replace(wrong_char, correct_char)
        texts.append(text)
        labels.append(label)
    return file_paths, texts, labels

SzegedDys_label_to_idx = {
    'CF014': 0,
    'CM013': 1,
    'F001': 2,
    'F002': 3,
    'F003': 4,
    'F004': 5,
    'M001': 6,
    'P01': 7,
    'P02': 8
}

def get_SzegedDys_as_list(wav_directory_path) :
    text_directory_path = wav_directory_path.parent / "text"
    texts = []
    labels = []
    file_paths = []
    for path_Laci_file_wav in wav_directory_path.iterdir():
        filename_array = path_Laci_file_wav.stem.split("_")
        label = SzegedDys_label_to_idx[filename_array[0]]
        text_name = "_"+filename_array[1]
        for path_Laci_file_text in text_directory_path.iterdir():
            if text_name in path_Laci_file_text.name :
                file_paths.append(str(path_Laci_file_wav))
                try :
                    with open(path_Laci_file_text, 'r', encoding='utf-8') as file:
                        text = str(file.read().rstrip('\n').capitalize())
                except UnicodeDecodeError :
                    with open(path_Laci_file_text, 'r', encoding='latin-1') as file:
                        text = str(file.read().rstrip('\n').capitalize())
                    char_fixes = {
                        'û': 'ű', 'õ': 'ő',
                        'ã': 'á', 'â': 'á', 'ê': 'é', 'î': 'í', 'ô': 'ó',
                        'Û': 'Ű', 'Õ': 'Ő',
                        'Ã': 'Á', 'Â': 'Á', 'Ê': 'É', 'Î': 'Í', 'Ô': 'Ó',
                    }
                    for wrong_char, correct_char in char_fixes.items():
                        text = text.replace(wrong_char, correct_char)
                texts.append(text)
                labels.append(label)
    return file_paths, texts, labels

Torgo_label_to_idx = {
    'F01': 0,
    'F03': 1,
    'F04': 2,
    'FC01': 3,
    'FC02': 4,
    'FC03': 5,
    'M01': 6,
    'M02': 7,
    'M03': 8,
    'M04': 9,
    'M05': 10,
    'MC01': 11,
    'MC02': 12,
    'MC03': 13,
    'MC04': 14
}

def get_TrogoGenerated_as_list(data_dir) :
    file_paths_dict = {}
    for path_TrogoGenerated_file in os.listdir(data_dir) :
        file_paths_dict[path_TrogoGenerated_file] = os.path.join(data_dir, path_TrogoGenerated_file)
    texts = []
    labels = []
    file_paths = []
    dataset = load_dataset("abnerh/TORGO-database")["train"]
    for element in dataset :
        if element['audio']['path'] in file_paths_dict :
            file_paths.append(file_paths_dict[element['audio']['path']])
            texts.append(element['transcription'].lower())
            uid = element["audio"]['path'].split("_")[0]
            labels.append(Torgo_label_to_idx[uid])
    return file_paths, texts, labels

def get_UASpeech_as_list(speakers, data_dir) :
    # The 15 dysarthric speakers ordered strictly by your data pipeline array
    all_dys_spks = [
        "F05", "M08", "M09", "M10", "M14",  # Very Low (5)
        "M05",  # Low
        "M11", "F04",  # M11 (High) | F04 (Low)
        "M07", "F02",  # M07 (Medium) | F02 (High)
        "M16",  # Medium
        "M04", "F03", "M12", "M01"  # High | High | High | High
    ]
    # The target integers matching Table 1 (5-2-2-6) and the paper's figures:
    # 1=Very Low, 2=Low, 3=Medium, 4=High
    all_dyslabels = [
        1, 1, 1, 1, 1,  # F05, M08, M09, M10, M14 -> Very Low
        2,  # M05                     -> Low
        4, 2,  # M11 -> High             |  F04 -> Low
        3, 4,  # M07 -> Medium           |  F02 -> High
        3,  # M16                     -> Medium
        4, 4, 4, 4  # M04, F03, M12, M01      -> High
    ]
    dys_speaker_dict = dict(zip(all_dys_spks, all_dyslabels))
    file_paths = []
    texts = []
    labels = []
    for speaker_folder in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_folder)
        if os.path.isdir(speaker_path):
            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                if file_name.endswith('.wav'):
                    if speaker_folder in speakers :
                        text_file_path = file_path.replace('.wav', '.sentence')
                        try:
                            with open(text_file_path, "r") as text_file:
                                word = text_file.readline().strip()  # Read the first line and strip newline/whitespace
                                file_paths.append(file_path)
                                texts.append(word)
                            if speaker_folder in dys_speaker_dict :
                                labels.append(dys_speaker_dict[speaker_folder])
                            else :
                                labels.append(0)
                        except FileNotFoundError :
                            pass
                elif os.path.isdir(file_path) and file_name in speakers:
                    for control_file_name in os.listdir(file_path) :
                        if control_file_name.endswith('.wav') :
                            control_file_path = os.path.join(file_path, control_file_name)
                            text_file_path = control_file_path.replace('.wav', '.sentence')
                            try:
                                with open(text_file_path, "r") as text_file:
                                    word = text_file.readline().strip()  # Read the first line and strip newline/whitespace
                                    file_paths.append(control_file_path)
                                    texts.append(word)
                                labels.append(0)
                            except FileNotFoundError :
                                pass
    return file_paths, texts, labels

def compute_metrics(predictions, references):
    wer = 100 * metric.compute(predictions=predictions, references=references)
    return {"wer": wer}

class UASpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap UASpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, speakers, data_dir, processor, tokenizer, torch_dtype, device):
        self.device = device
        self.processor = processor
        self.tokenizer = tokenizer
        self.torch_dtype = torch_dtype
        self.all_dys_spks = [
            "F05", "M08", "M09", "M10", "M14",  # Very Low Severity (Label 1)
            "M05",  # Medium Severity (Label 3)
            "M11", "F04",  # Low Severity (Label 2)
            "M07", "F02",  # High Severity (Label 4)
            "M16",  # Medium Severity (Label 3)
            "M04", "F03", "M12", "M01"  # High/Very Low mix
        ]
        self.all_dyslabels = [
            1, 1, 1, 1, 1,  # F05, M08, M09, M10, M14 -> Very Low (1)
            3,  # M05                     -> Medium (3)
            2, 2,  # M11, F04                -> Low (2)
            4, 4,  # M07, F02                -> High (4)
            3,  # M16                     -> Medium (3)
            4, 4, 1, 4  # M04 (High), F03 (High), M12 (Very Low), M01 (High)
        ]
        self.dys_speaker_dict = dict(zip(self.all_dys_spks, self.all_dyslabels))
        self.file_paths = []
        self.labels = []
        for speaker_folder in os.listdir(data_dir):
            speaker_path = os.path.join(data_dir, speaker_folder)
            if os.path.isdir(speaker_path):
                for file_name in os.listdir(speaker_path):
                    file_path = os.path.join(speaker_path, file_name)
                    if file_name.endswith('.wav'):
                        if speaker_folder in speakers :
                            text_file_path = file_path.replace('.wav', '.txt')
                            try:
                                with open(text_file_path, "r") as text_file:
                                    word = text_file.readline().strip()  # Read the first line and strip newline/whitespace
                                    self.file_paths.append((file_path, word))
                                if speaker_folder in self.dys_speaker_dict :
                                    self.labels.append(self.dys_speaker_dict[speaker_folder])
                                else :
                                    self.labels.append(0)
                            except FileNotFoundError :
                                pass
                    elif os.path.isdir(file_path) and file_name in speakers:
                        for control_file_name in os.listdir(file_path) :
                            if control_file_name.endswith('.wav') :
                                control_file_path = os.path.join(file_path, control_file_name)
                                text_file_path = control_file_path.replace('.wav', '.txt')
                                try:
                                    with open(text_file_path, "r") as text_file:
                                        word = text_file.readline().strip()  # Read the first line and strip newline/whitespace
                                        self.file_paths.append((control_file_path, word))
                                    self.labels.append(0)
                                except FileNotFoundError :
                                    pass

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        file_path, text = self.file_paths[item]

        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        assert sample_rate == 16000
        sample = {'path:': file_path,
                      'array': waveform.flatten().numpy(),
                      'sampling_rate' : 16000}
        input_features = self.processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt", torch_dtype=self.torch_dtype).input_features
        input_features = input_features.half().to(self.device)
        text_tokenized = self.tokenizer(text).input_ids
        decoded_text = self.tokenizer.decode(text_tokenized, skip_special_tokens=True)
        #audio = pad_or_trim(waveform.flatten()).to(self.device)
        #mel = log_mel_spectrogram(audio)
        label = self.labels[item]

        return input_features, decoded_text, label

def get_hirado_file_paths(data_dir) :
    file_paths = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            text_file_name = file_name.replace('.wav', '.txt')
            text_file_path = os.path.join(data_dir, text_file_name)
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(text_file_path, "r") as text_file:
                    word = text_file.read().strip().replace('\n', ' ')  # Read file and strip newline/whitespace
                    file_paths.append((file_path, word))
            except FileNotFoundError:
                pass
    return file_paths
class Hirado(torch.utils.data.Dataset):
    """
    A simple class to wrap Hirado.
    """
    def __init__(self, data_dir, processor, tokenizer, torch_dtype, device):
        self.device = device
        self.processor = processor
        self.tokenizer = tokenizer
        self.torch_dtype = torch_dtype
        self.file_paths = get_hirado_file_paths(data_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        file_path, text = self.file_paths[item]

        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)
        assert sample_rate == 16000
        sample = {'path:': file_path,
                      'array': waveform.flatten().numpy(),
                      'sampling_rate' : 16000}
        input_features = self.processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt", torch_dtype=self.torch_dtype).input_features
        input_features = input_features.half().to(self.device)
        text_tokenized = self.tokenizer(text).input_ids
        decoded_text = self.tokenizer.decode(text_tokenized, skip_special_tokens=True)
        #audio = pad_or_trim(waveform.flatten()).to(self.device)
        #mel = log_mel_spectrogram(audio)
        label = torch.empty(0)

        return input_features, decoded_text, label

