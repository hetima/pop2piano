#!/usr/bin/env python3

import os
import sys
from InquirerPy import inquirer
from InquirerPy.validator import NumberValidator
import subprocess

MODEL_FILE_NAME = "model-1999-val_0.67311615.ckpt"

# Windowsのパスだったら/mnt/...に変換
def path_to_mnt(str):
    if len(str) <= 3:
        return str
    if str[0] == '"' or str[0] == "'":
        str = str[1:-1]
    if str[1] == ':':
        str = str.replace('\\', '/')
        str = '/mnt/' + str[0].lower() + str[2:]
    return str

def model_file_path():
    thisdir = os.path.dirname(__file__)
    model_file = os.path.join(thisdir, MODEL_FILE_NAME)
    if not os.path.exists(model_file): model_file = os.path.join(os.path.dirname(thisdir), MODEL_FILE_NAME)
    # if not os.path.exists(model_file): model_file = model_file = os.path.join(thisdir, "models", MODEL_FILE_NAME)
    return model_file

def pop2piano_main(audio_file, composer, bpm):
    import os
    import sys
    sys.path.append("pop2piano")

    import torch
    from omegaconf import OmegaConf
    import note_seq
    from transformer_wrapper import TransformerWrapper

    model_file = model_file_path()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load("pop2piano/config.yaml")
    wrapper = TransformerWrapper(config)
    wrapper = wrapper.load_from_checkpoint(model_file, config=config).to(device)
    model = "dpipqxiy"
    wrapper.eval()

    composer_num = int(composer) - 1
    composers = [
        'composer1', 'composer2', 'composer3', 'composer4', 'composer5',
        'composer6', 'composer7', 'composer8', 'composer9', 'composer10',
        'composer11', 'composer12', 'composer13', 'composer14', 'composer15',
        'composer16', 'composer17', 'composer18', 'composer19', 'composer20', 'composer21'
    ]
    if len(composers) <= composer_num:
        composer_num = 0

    midi_bpm = float(bpm)
    pm, composer, mix_path, midi_path = wrapper.generate(
        audio_path=audio_file,
        composer=composers[composer_num],
        model=model,
        show_plot=False,
        save_midi=True,
        save_mix=False,
        midi_bpm=midi_bpm,
    )
    print(midi_path)


def do_pop2piano(args):
    if args.audio_file != "":
        src_path = path_to_mnt(args.audio_file)
        print('audio_file = ' + src_path)
    else:
        src_path = inquirer.filepath(message="audio file path:", filter=path_to_mnt).execute()
    if not os.path.exists(src_path):
        print("file not found")
        return
    if os.path.isdir(src_path):
        print("target is folder")
        return
    if args.composer != 1 or "--composer" in sys.argv or "--composer=1" in sys.argv:
        composer = args.composer
        print('composer = ' + str(composer))
    else:
        composer = inquirer.text(message="select composer (1-21):", default="1", validate=NumberValidator()).execute()
    if args.bpm != 120.0 or "--bpm" in sys.argv or "--bpm=120" in sys.argv:
        bpm = args.bpm
        print('MIDI file BPM = ' + str(bpm))
    else:
        bpm = inquirer.text(message="output file bpm:", default="120").execute()
    pop2piano_main(src_path, composer, bpm)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="pop2piano midi exporter")
    parser.add_argument("audio_file", type=str, default="", help="audio file path", nargs='?')
    parser.add_argument("--composer", type=int, default=1, help="composer type (1-21)")
    parser.add_argument("--bpm", type=float, default=120.0, help="output MIDI file BPM")
    args = parser.parse_args()
    do_pop2piano(args)
