import math
from abc import ABCMeta, abstractmethod
from enum import Enum

import cv2
import numpy as np
import system_manager as sm
from PIL import Image

config = sm.get_config()


class Effect(metaclass=ABCMeta):

    rem = 0.0

    class EffectType(Enum):
        NORMAL = 1
        FADE_IN = 2
        RADIAL_BLUR = 3
        LINE_MOVE = 4
        SLIDE = 5

    def __init__(self, meta, total_time, e_type):
        self.i = 0
        self.video = meta['video']
        self.f_time = meta['f_time']
        self.total_f_count = int(total_time / self.f_time)
        self.total_time = total_time
        self.e_type = e_type

    @abstractmethod
    def mk_image(self):
        pass

    def apply(self):
        default_count = self.total_f_count
        while self.i < self.total_f_count:

            rem_count = int(Effect.rem / self.f_time)
            if rem_count != 0.0:
                self.total_f_count += rem_count
                Effect.rem -= self.f_time * rem_count

            if self.i == default_count - 1:
                Effect.rem += self.total_time % (default_count * self.f_time)

            self.video.write(self.mk_image())
            self.i += 1
        return self.total_f_count


class Normal(Effect):

    def __init__(self, meta, img, total_time):
        super().__init__(meta, total_time, self.EffectType.NORMAL)
        self.img = img

    def mk_image(self):
        return self.img


class FadeIn(Effect):

    def __init__(self, meta, img1, img2, in_time, total_time):
        super().__init__(meta, total_time, self.EffectType.FADE_IN)
        back = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        back = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
        back = Image.fromarray(back)
        self.back = back.convert('RGBA')

        over = cv2.cvtColor(img2, cv2.COLOR_BGRA2RGBA)
        over = Image.fromarray(over)
        self.over = over.convert('RGBA')

        self.alpha = 0.0
        self.in_f_count = int(in_time / self.f_time)

    def mk_image(self):
        if self.i < self.in_f_count:
            self.alpha += 255 / self.in_f_count
            self.over.putalpha(int(self.alpha))
            frame = Image.alpha_composite(self.back, self.over)
            frame = np.asarray(frame)
        else:
            frame = np.asarray(self.over)

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class RadialBlur(Effect):

    def __init__(self, meta, img1, img2, in_time, total_time, min_ratio, iterations, margin):
        super().__init__(meta, total_time, self.EffectType.RADIAL_BLUR)
        self.img1 = img1
        self.img2 = img2
        self.in_f_count = int(in_time / self.f_time)
        self.ratio = 1.0
        self.change = (1.0 - min_ratio) / self.in_f_count
        self.iterations = iterations
        self.margin = margin

    def mk_image(self):
        if self.i < self.in_f_count:
            self.ratio -= self.change
            pos = (self.img1.shape[1] // 2, self.img1.shape[0] // 2)
            frame = self.radial_blur_frame(self.img1, pos)
        else:
            frame = self.img2

        return frame

    def radial_blur_frame(self, src, pos):
        h, w = src.shape[0:2]
        n = self.iterations
        m = self.margin

        bg = np.ones(src.shape, dtype=np.uint8) * 0
        bg = cv2.resize(bg, (int(m * w), int(m * h)))

        bg[int((m - 1) * h / 2):int((m - 1) * h / 2) + h, int((m - 1) * w / 2):int((m - 1) * w / 2) + w] = src

        image_list = []
        h *= m
        w *= m
        c_x = pos[0] * m
        c_y = pos[1] * m

        for i in range(n):
            r = self.ratio + (1 - self.ratio) * (i + 1) / n
            shrunk = cv2.resize(src, (int(r * w), int(r * h)))
            left = int((1 - r) * c_x)
            right = left + shrunk.shape[1]
            top = int((1 - r) * c_y)
            bottom = top + shrunk.shape[0]
            bg[top:bottom, left:right] = shrunk
            image_list.append(bg.astype(np.int32))

        dst = sum(image_list) / n
        dst = dst.astype(np.uint8)

        r = (1 + self.ratio) / 2
        dst = dst[int((1 - r) * c_y):int(((1 - r) * c_y + h) * r), int((1 - r) * c_x):int(((1 - r) * c_x + w) * r)]
        dst = cv2.resize(dst, (int(w / m), int(h / m)))
        return dst


class LineMove(Effect):

    def __init__(self, meta, img, total_time, ratio, from_pos, to_pos):
        super().__init__(meta, total_time, self.EffectType.LINE_MOVE)
        self.img = img
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.from_x = from_pos[0]
        self.from_y = from_pos[1]
        self.to_x = to_pos[0]
        self.to_y = to_pos[1]

        area = self.height * self.width * ratio
        tan = self.height / self.width
        self.a = math.sqrt(area / tan)
        self.b = self.a * tan

        self.dis, self.angle = self.get_line(
            self.from_x, self.from_y,
            self.to_x, self.to_y,
            self.a, self.b)

        self.dis_per_frame = self.dis / self.total_f_count
        self.dc = self.dis_per_frame * math.cos(self.angle)
        self.ds = self.dis_per_frame * math.sin(self.angle)

    def mk_image(self):
        x1 = 0
        y1 = 0
        if self.from_y == self.to_y:
            y1 = self.from_y
            x1 = self.i * self.dis_per_frame + self.from_x if self.from_x < self.to_x else \
                (self.total_f_count - self.i - 1) * self.dis_per_frame + self.to_x if self.from_x > self.to_x \
                else 0
        elif self.from_x == self.to_x:
            x1 = self.from_x
            y1 = self.i * self.dis_per_frame + self.from_y if self.from_y < self.to_y else \
                (self.total_f_count - self.i - 1) * self.dis_per_frame + self.to_y if self.from_y > self.to_y \
                else 0
        if self.from_x < self.to_x:
            if self.from_y < self.to_y:
                x1 = self.i * self.dc + self.from_x
                y1 = self.i * self.ds + self.from_y
            elif self.from_y > self.to_y:
                x1 = self.i * self.dc + self.from_x
                y1 = (self.total_f_count - self.i - 1) * self.ds + self.to_y
        elif self.from_x > self.to_x:
            if self.from_y < self.to_y:
                x1 = -(self.i * self.dc) + self.from_x - self.a
                y1 = self.i * self.ds + self.from_y
            elif self.from_y > self.to_y:
                x1 = -((self.total_f_count - self.i - 1) * self.dc) + self.from_x - self.a
                y1 = (self.total_f_count - self.i - 1) * self.ds + self.to_y

        x2 = x1 + self.a
        y2 = y1 + self.b

        x1, x2 = self.__fix(x1, x2, self.width, self.a)
        y1, y2 = self.__fix(y1, y2, self.height, self.b)

        cut = self.img[int(y1): int(y2), int(x1): int(x2)]
        cut = cv2.resize(cut, (self.width, self.height))

        return cut

    @staticmethod
    def get_line(x1, y1, x2, y2, a, b):
        angle = math.atan((y2 - y1) / (x2 - x1)) if x1 != x2 and y1 != y2 else 0

        if (x1 < x2 and y1 > y2) or (x1 > x2 and y1 < y2):
            angle *= -1

        x2 += -a if x1 < x2 else a if x1 > x2 else 0
        y2 += -b if y1 < y2 else b if y1 > y2 else 0

        dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return dis, angle

    @staticmethod
    def __fix(val1, val2, max_val, dif):
        if val1 < 0:
            val1 = 0
            val2 = dif
        elif val1 > max_val:
            val1 = max_val
            val2 = max_val - dif

        if val2 < 0:
            val2 = 0
            val1 = dif
        elif val2 > max_val:
            val2 = max_val
            val1 = max_val - dif

        return val1, val2


class Slide(Effect):

    def __init__(self, meta, img, total_time):
        super().__init__(meta, total_time, self.EffectType.SLIDE)
        self.img = img
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.dis = self.height // self.total_f_count * 2

        self.offset_y = 0

    def mk_image(self):
        back = np.full((self.height, self.width, 3), 0, dtype=np.uint8)

        if self.offset_y > self.height:
            cut1 = self.img[0: self.height - (self.offset_y - self.height), 0: self.width // 2]
            cut2 = self.img[self.offset_y - self.height: self.height, self.width // 2: self.width]
            back[self.height - cut1.shape[0]: self.height, 0: cut1.shape[1]] = cut1
            back[0: cut2.shape[0], cut1.shape[1]: cut1.shape[1] + cut2.shape[1]] = cut2
        else:
            cut1 = self.img[self.height - self.offset_y: self.height, 0: self.width // 2]
            cut2 = self.img[0: self.offset_y, self.width // 2: self.width]
            back[self.offset_y - cut1.shape[0]: self.offset_y, 0: cut1.shape[1]] = cut1
            back[self.height - self.offset_y: self.height - (self.offset_y - cut2.shape[0]),
            cut1.shape[1]: cut1.shape[1] + cut2.shape[1]] = cut2

        self.offset_y += self.dis

        return back


def apply_effect(meta, width, height, tempo, ignore_time, img_list, i, total_count, index):
    start = ignore_time[0]
    end = ignore_time[len(ignore_time) - 1]
    back = np.full((height, width, 3), 0, dtype=np.uint8)
    img = img_list[index] if index < len(img_list) else back
    f_time = meta['f_time']

    if i >= total_count:
        return

    start_time = to_tempo(tempo, start[1])
    if index == 0 and start[0] // tempo == 0.0 and 0.0 < start[1] - start[0]:
        i += FadeIn(meta, back, img, start_time, start_time).apply() - 1
        apply_effect(meta, width, height, tempo, ignore_time, img_list, i, total_count, index + 1)
        return

    end1 = to_tempo(tempo, end[0])
    end2 = to_tempo(tempo, end[1])
    end_time = end2 - end1
    if end1 <= (i + 1) * f_time and (total_count * f_time - end_time) - end1 <= tempo:
        FadeIn(meta, img, back, end_time, end_time).apply()
        return

    r = int(np.random.rand() * 100)

    scope = 'effect_chance'
    fade_in = config[scope]['fade_in']
    radial_blur = config[scope]['radial_blur'] + fade_in
    slide = config[scope]['slide'] + radial_blur
    line_move = config[scope]['line_move'] + slide
    fc_mag = config['frame_count_mag']

    total_time = tempo * fc_mag

    if 0 <= r < fade_in and 0 < index < len(img_list):
        effect = FadeIn(meta, img_list[index - 1], img, total_time / 2, total_time)
    elif fade_in <= r < radial_blur and 0 < index < len(img_list):
        effect = RadialBlur(meta, img_list[index - 1], img, 0.25, total_time, 0.5, 20, 1.3)
    elif radial_blur <= r < slide:
        effect = Slide(meta, img, total_time * 4)
    elif slide <= r < line_move:
        pos_x = [0, img.shape[1]]
        pos_y = [0, img.shape[0], img.shape[0] // 2]

        from_pos = [0] * 2
        from_pos[0] = pos_x[int(np.random.rand() * len(pos_x))]
        from_pos[1] = pos_y[int(np.random.rand() * len(pos_y))]

        to_pos = [0] * 2
        to_pos[0] = pos_x[0 if from_pos[0] == pos_x[1] else 1]
        to_pos[1] = pos_y[2 if from_pos[1] == pos_y[2] else 0 if from_pos[1] == pos_y[1] else 1]

        effect = LineMove(meta, img, total_time, 0.75, from_pos, to_pos)
    else:
        effect = Normal(meta, img, total_time)

    c = effect.apply()
    i += c
    print(f'Effect: {effect.e_type.name}')
    print(f'Used frame: {c}')
    del effect

    if index == 0:
        i -= 1
    apply_effect(meta, width, height, tempo, ignore_time, img_list, i, total_count, index + 1)


def to_tempo(tempo, time):
    return tempo * (time // tempo)
