import torch

torch_dtype = torch.float16
GPU = 0
SAMPLING_RATE = 16000
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
CHUNK_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

ALL_SPEAKERS = "all_spks"
ALL_DYSARTHRIC_SPEAKERS = "all_dys_spks"
TRAIN_SPEAKERS = "train_spks"
TRAIN_CONTROL_SPEAKERS = "train_cspks"
TEST_SPEAKERS = "test_spks"
TEST_CONTROL_SPEAKERS = "test_cspks"
TEST_DYSARTHRIC_SPEAKERS = "test_dspks"

UASPEECH = "UASpeech"
HIRADO = "hirado"
TORGO = "Torgo"
LJSPEECH = "LJSpeech"
LACICON = "LaciControl"
LACIDYS = "LaciDys"
SZEGEDYS = "SzegedDys"
HUNDYS = "HunDys"
SZINDBAD = "Szindbad"
BEA = "bea"

speakers_dict = {
    ALL_SPEAKERS: ["M09", "M14", "M10", "M08", "F05", "M05", "M11", "F04", "M07", "F02", "M16", "M04", "F03", "M12", "M01",
                 "CF02", "CF03", "CF04", "CF05", "CM01", "CM04", "CM05", "CM06", "CM08", "CM09", "CM10", "CM12", "CM13"],
    TRAIN_SPEAKERS: ['F02', 'F03', 'F04', 'F05', 'CM01', 'CF02', 'M11', 'M12', 'M14', 'M16'] }
speakers_dict[TRAIN_CONTROL_SPEAKERS] = [spk for spk in speakers_dict[ALL_SPEAKERS] if spk.startswith("C") and spk in speakers_dict[TRAIN_SPEAKERS]]
speakers_dict[ALL_DYSARTHRIC_SPEAKERS] = [spk for spk in speakers_dict[ALL_SPEAKERS] if not spk.startswith("C")]
speakers_dict[TEST_SPEAKERS] = [item for item in speakers_dict[ALL_SPEAKERS] if item not in speakers_dict[TRAIN_SPEAKERS]]
speakers_dict[TEST_CONTROL_SPEAKERS] = [spk for spk in speakers_dict[ALL_SPEAKERS] if spk.startswith("C") and spk not in speakers_dict[TRAIN_SPEAKERS]]
speakers_dict[TEST_DYSARTHRIC_SPEAKERS] = [spk for spk in speakers_dict[ALL_SPEAKERS] if not spk.startswith("C") and spk not in speakers_dict[TRAIN_SPEAKERS]]

dataset = SZEGEDYS
#dataset = "hirado"
hungarian_datasets = {HIRADO, LACICON, LACIDYS, SZEGEDYS, HUNDYS, SZINDBAD, BEA}
if dataset in hungarian_datasets :
    lang = "hu"
else :
    lang = "en"

uaspeech_dir = "/home/berta/data/UASpeech/audio"
#torgo_dir = "/home/berta/data/Torgo/knnvc-generated/"
torgo_dir = "/home/berta/data/Torgo/urhythmic-fine-generated/"
laci_control_dir = "/home/berta/data/LaciControl2018/"
szindbad_dir = "/home/berta/data/Szindbad/"
bea_dir = "/home/berta/data/bea_wav_original/"
laci_dys_dir = "/home/berta/data/LaciDys2025/"
hundys_dir = "/home/berta/data/HungarianDysartriaDatabase/wav"
#laci_dys_dir = "LaciDys"
processed_paths = {}
for key in speakers_dict :
    processed_paths[key] = "/home/berta/data/" + dataset + "/processed_" + key + ".pt"
processed_paths[TRAIN_SPEAKERS] = "/home/berta/data/" + dataset + "/processed_train.pt"
processed_paths[TEST_SPEAKERS] = "/home/berta/data/" + dataset + "/processed_test.pt"
processed_paths[TEST_DYSARTHRIC_SPEAKERS] = "/home/berta/data/" + dataset + "/processed_test_dspks.pt"

hirado_dir = "/home/berta/data/hirado/"
whisper_arch = "openai/whisper-large-v3"
#whisper_arch = "openai/whisper-large-v2"
#whisper_arch = "openai/whisper-medium"
# TORGO class as UASpeech
#TORGO_dys_spks = ["F03", "F04", "M03", "F01", "M05", "M01", "M02", "M04"]
#TORGO_dyslabels = [1, 1, 1, 2, 2, 3, 3, 3]
# TORGO class as RnV2025 paper told
TORGO_dys_spks = ["F04", "M03", "F03", "M05", "F01", "M01", "M02", "M04"]
TORGO_dyslabels = [1, 1, 2, 3, 4, 4, 4, 4]

TORGO_dys_speaker_dict = dict(zip(TORGO_dys_spks, TORGO_dyslabels))
label_count = {TORGO: 15, UASPEECH: 5, LACICON: 1, LACIDYS: 5, HUNDYS: 50, SZEGEDYS: 9}