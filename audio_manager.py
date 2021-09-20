import wave

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import system_manager


class AudioManager:

    def __init__(self, path):
        self.path = path

    def get_bpm(self):
        duration = 30
        x_sr = 200
        bpm_min, bpm_max = 60, 240

        # 楽曲の信号を読み込む
        y, sr = librosa.load(self.path, offset=38, duration=duration, mono=True)

        # ビート検出用信号の生成
        # リサンプリング & パワー信号の抽出
        x = np.abs(librosa.resample(y, sr, x_sr)) ** 2
        x_len = len(x)

        # 各BPMに対応する複素正弦波行列を生成
        M = np.zeros((bpm_max, x_len), dtype=complex)
        for bpm in range(bpm_min, bpm_max):
            thete = 2 * np.pi * (bpm / 60) * (np.arange(0, x_len) / x_sr)
            M[bpm] = np.exp(-1j * thete)

        # 各BPMとのマッチング度合い計算
        # （複素正弦波行列とビート検出用信号との内積）
        x_bpm = np.abs(np.dot(M, x))

        # BPM　を算出
        bpm = np.argmax(x_bpm)
        return bpm

    def length(self):
        wf = wave.open(self.path, mode='rb')
        return wf.getnframes() / wf.getframerate()

    @staticmethod
    def labels_list_to_df(labels_list):
        labels_01_list = []

        for i in range(len(labels_list)):
            labels_01_list.append(np.bincount([labels_list[i]]))

        return pd.DataFrame(labels_01_list)

    @staticmethod
    def df_to_df_list(df):
        df_list = []
        for i in range(len(df.columns)):
            df_list.append(df[i])

        return df_list

    def get_ignore_time(self):
        data, samplerate = sf.read(self.path)
        config = system_manager.get_config()['ignore_sound_range']

        thres = config['thres']
        amp = np.abs(data)
        b = amp > thres

        min_silence_duration = config['min_silence_duration']

        silences = []
        prev = 0
        entered = 0

        for i, v in enumerate(b):
            t = type(prev)
            if (t is int and prev == 1) or (t is np.ndarray and prev.any()) \
                    and not v.all():  # enter silence
                entered = i
            if (t is int and prev == 0) or (t is np.ndarray and not prev.all()) \
                    and v.any():  # exit silence
                duration = (i - entered) / samplerate
                if duration > min_silence_duration:
                    silences.append([entered / samplerate, i / samplerate])
                    entered = 0
            prev = v

        if 0 < entered < len(b):
            silences.append([entered / samplerate, len(b) / samplerate])

        return silences
