import json
import os

dir_path = './data'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


def write(data, path):
    f = open(path, 'w')
    json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))


def init():
    json_path = './data/config.json'
    if os.path.exists(json_path):
        return

    data = {
        'effect_chance': {
            'fade_in': 12,
            'radial_blur': 5,
            'slide': 2,
            'line_move': 10
        },
        'ignore_sound_range': {
            'thres': 0.15,
            'min_silence_duration': 0.2
        },
        'frame_count_mag': 1
    }

    write(data, json_path)
    print(f'Made a config file({json_path}).')


def get_config():
    return json.load(open('./data/config.json', 'r'))


def log(key, val):
    print(f'{key} = {val}')
