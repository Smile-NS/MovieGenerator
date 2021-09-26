# -*- coding: utf-8 -*-
import json
import math
import os
import tkinter
from tkinter import filedialog

import cv2
import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip

import system_manager as sm
import video_effect
from audio_manager import AudioManager


# pipenv run pyinstaller --onefile main.py --additional-hooks=extra-hooks


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def mk_video(width, height, fps, audio, images, output):
    save_data = {
        'width': width,
        'height': height,
        'fps': fps,
        'audio_path': audio.path,
        'images_path': images,
        'output_path': output
    }

    t = output.split('/')
    video_name = t[len(t) - 1]
    output_dir = output[0:len(output) - len(video_name) - 1]
    sm.write(save_data, f'{output_dir}/{video_name}.json')

    video_time = audio.length()
    sm.log('time', f'{video_time}(sec)')

    print('Start loading images...')
    img_list = []
    for filename in find_all_files(images):
        img = imread(filename)

        ratio = (width * height) / (img.shape[1] * img.shape[0])
        img_width = img.shape[1] * math.sqrt(ratio)
        img_height = img.shape[0] * math.sqrt(ratio)
        ex_rate = width / img_width if img_width < width else height / img_height if img_height < height else 1
        img_width *= ex_rate
        img_height *= ex_rate
        margin_h = (img_height - height) / 2
        margin_w = (img_width - width) / 2

        img = cv2.resize(img, (int(img_width), int(img_height)))
        img = img[
              int(margin_h): int(img_height - margin_h),
              0: width]

        img_list.append(img)
        fn = filename.replace('\\', '/')
        print(f'Loaded: {fn}')

    print(f'Loading all images is complete! ({len(img_list)})')

    temp = f'{output_dir}/temp_{video_name}.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(temp, fourcc, fps, (width, height))

    total_count = int(video_time * fps)
    frame_time = 1.0 / fps
    bpm = audio.get_bpm()
    tempo = 60 / bpm

    sm.log('total frame count', total_count)
    sm.log('frame time', f'{frame_time}(sec)')
    sm.log('bpm', bpm)
    sm.log('tempo', f'{tempo}(sec)')

    ignore_time = audio.get_ignore_time()
    print('Start to apply effects to the pictures...')

    meta = {
        'video': video,
        'f_time': frame_time
    }

    video_effect.apply_effect(meta, width, height, tempo, ignore_time, img_list, 0, total_count, 0)
    print('Finished to apply effects.')

    video.release()
    print(f'Made a temp video file({temp}), do not delete it, please.')

    video = VideoFileClip(temp)
    video = video.set_audio(AudioFileClip(audio.path))
    video.write_videofile(output)
    print(f'Making a video({output}) is complete!')
    os.remove(temp)
    print('Removed the temp video file.')
    del audio
    print('Finished all processing.')


ext_list = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']


def find_all_files(folder):
    file_list = []
    for curDir, dirs, files in os.walk(folder):
        for a_file in files:
            path = f'{curDir}\\{a_file}'
            base, ext = os.path.splitext(f'{curDir}\\{a_file}')
            if ext in ext_list:
                file_list.append(path)
    return file_list


size = '444x250'


def menu():
    tki = tkinter.Tk()
    tki.geometry(size)
    tki.title('MENU - MovieGenerator')

    btn1 = tkinter.Button(tki, text='New project', command=lambda: new_project(tki), width=20)
    btn1.pack()

    btn2 = tkinter.Button(tki, text='Import project', width=20,
                          command=lambda: import_action(filedialog.askopenfilename(
                              title="Select a save data",
                              filetypes=[("Save data", ".json")],
                              initialdir="./"
                          )))
    btn2.pack()

    def import_action(path):
        data = json.load(open(path, 'r'))
        create(tki, str(data['width']), str(data['height']), str(data['fps']),
               str(data['audio_path']), str(data['images_path']), str(data['output_path']))

    tki.resizable(False, False)
    tki.mainloop()


def new_project(menu_ui):
    menu_ui.destroy()
    tki = tkinter.Tk()
    tki.geometry(size)
    tki.title('New project - MovieGenerator')

    wl = tkinter.Label(text='width')
    we = tkinter.Entry(width=20)

    hl = tkinter.Label(text='height')
    he = tkinter.Entry(width=20)

    fl = tkinter.Label(text='fps')
    fe = tkinter.Entry(width=10)

    al = tkinter.Label(text='audio')
    ae = tkinter.Entry(width=30)
    ab = tkinter.Button(tki, text='…', width=3,
                        command=lambda: ae.insert(tkinter.END, filedialog.askopenfilename(
                            title="Select a Audio file",
                            filetypes=[("Audio file", ".wav")],
                            initialdir="./"
                        )))
    ab.place(x=290, y=87)

    dl = tkinter.Label(text='images dir')
    de = tkinter.Entry(width=30)
    db = tkinter.Button(tki, text='…', width=3,
                        command=lambda: de.insert(tkinter.END, filedialog.askdirectory(
                            title="Select an images directory",
                            initialdir="./"
                        )))
    db.place(x=290, y=117)

    ol = tkinter.Label(text='output')
    oe = tkinter.Entry(width=30)
    ob = tkinter.Button(tki, text='…', width=3,
                        command=lambda: oe.insert(tkinter.END, filedialog.asksaveasfilename(
                            defaultextension='mp4',
                            title="Save",
                            filetypes=[("Video file", "*.mp4")],
                            initialdir="./"
                        )))
    ob.place(x=290, y=147)

    cb = tkinter.Button(tki, text='create',
                        command=lambda: create(tki, we.get(), he.get(), fe.get(), ae.get(), de.get(), oe.get()))
    cb.place(x=30, y=200)

    for i, label in enumerate([wl, hl, fl, al, dl, ol]):
        label.place(x=30, y=i * 30)

    for i, enter in enumerate([we, he, fe, ae, de, oe]):
        enter.place(x=100, y=i * 30)

    tki.resizable(False, False)
    tki.mainloop()


def create(tki, width, height, fps, audio, images, output):
    t = output.split('/')
    output_dir = output[0:len(output) - len(t[len(t) - 1]) - 1]

    if not width.isdecimal():
        print('width: Invalid value')
        return
    if not height.isdecimal():
        print('height: Invalid value')
        return
    if not fps.isdecimal():
        print('fps: Invalid value')
        return
    if not os.path.exists(audio):
        print('sound file: Invalid location')
        return
    if not os.path.exists(images):
        print('images dir: Invalid location')
        return
    if not os.path.exists(output_dir):
        print('output dir: Invalid location')
        return

    tki.destroy()

    width = int(width)
    height = int(height)
    fps = int(fps)

    sm.log('width', width)
    sm.log('height', height)
    sm.log('fps', fps)
    sm.log('audio', audio)
    sm.log('images', images)
    sm.log('output', output)

    mk_video(width, height, fps, AudioManager(audio), images, output)


print('Start processing...')
sm.init()
menu()
