import math

import cv2
import numpy as np
import system_manager as sm
from PIL import Image


class EffectManager:

    def __init__(self, video, tempo, frame_time):
        self.video = video
        self.tempo = tempo
        self.frame_time = frame_time
        self.__frame_count = tempo // frame_time
        self.__rem = 0

    def fade_in(self, img1, img2, in_f_count, total_f_count):
        back = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        back = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
        back = Image.fromarray(back)
        back = back.convert('RGBA')

        over = cv2.cvtColor(img2, cv2.COLOR_BGRA2RGBA)
        over = Image.fromarray(over)
        over = over.convert('RGBA')

        alpha = 0
        i = 0
        while i < total_f_count:

            if self.__rem // self.frame_time != 0:
                total_f_count += 1
                self.__rem -= self.frame_time

            if i < in_f_count:
                alpha += 255 / in_f_count
                over.putalpha(int(alpha))
                frame = Image.alpha_composite(back, over)
                frame = np.asarray(frame)
            else:
                frame = np.asarray(over)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if i == self.__frame_count - 1:
                self.__rem += self.tempo - (self.frame_time * self.__frame_count)

            self.video.write(frame)
            i += 1

        return total_f_count

    def normal(self, img, f_count):
        i = 0
        while i < f_count:

            if self.__rem // self.frame_time != 0:
                f_count += 1
                self.__rem -= self.frame_time

            if i == self.__frame_count - 1:
                self.__rem += self.tempo - (self.frame_time * self.__frame_count)

            self.video.write(img)
            i += 1

        return f_count

    @staticmethod
    def __radial_blur_frame(src, pos, ratio, iterations, margin):
        h, w = src.shape[0:2]
        n = iterations
        m = margin

        # 背景を作成する．お好みで255を0にすると黒背景にできる．
        bg = np.ones(src.shape, dtype=np.uint8) * 255
        bg = cv2.resize(bg, (int(m * w), int(m * h)))

        # 背景の中心に元画像を配置
        bg[int((m - 1) * h / 2):int((m - 1) * h / 2) + h, int((m - 1) * w / 2):int((m - 1) * w / 2) + w] = src

        image_list = []
        h *= m
        w *= m
        c_x = pos[0] * m
        c_y = pos[1] * m

        # 縮小画像の作成
        for i in range(n):
            r = ratio + (1 - ratio) * (i + 1) / n
            shrunk = cv2.resize(src, (int(r * w), int(r * h)))
            left = int((1 - r) * c_x)
            right = left + shrunk.shape[1]
            top = int((1 - r) * c_y)
            bottom = top + shrunk.shape[0]
            bg[top:bottom, left:right] = shrunk
            image_list.append(bg.astype(np.int32))

        # 最終的な出力画像の作成
        dst = sum(image_list) / n
        dst = dst.astype(np.uint8)

        r = (1 + ratio) / 2
        dst = dst[int((1 - r) * c_y):int(((1 - r) * c_y + h) * r), int((1 - r) * c_x):int(((1 - r) * c_x + w) * r)]
        dst = cv2.resize(dst, (int(w / m), int(h / m)))
        return dst

    def radial_blur(self, img1, img2, in_f_count, total_f_count, min_ratio, iterations, margin):
        i = 0
        ratio = 1.0
        change = (1.0 - min_ratio) / in_f_count
        while i < total_f_count:

            if self.__rem // self.frame_time != 0:
                total_f_count += 1
                self.__rem -= self.frame_time

            if i < in_f_count:
                ratio -= change
                pos = (img1.shape[1] // 2, img1.shape[0] // 2)
                frame = self.__radial_blur_frame(img1, pos, ratio, iterations, margin)
            else:
                frame = img2

            if i == self.__frame_count - 1:
                self.__rem += self.tempo - (self.frame_time * self.__frame_count)

            self.video.write(frame)
            i += 1

        return total_f_count

    def line_move(self, img, f_count, ratio, from_pos, to_pos):
        width = img.shape[1]
        height = img.shape[0]
        from_x = from_pos[0]
        from_y = from_pos[1]
        to_x = to_pos[0]
        to_y = to_pos[1]

        area = height * width * ratio
        tan = height / width
        a = math.sqrt(area / tan)
        b = a * tan

        dis, angle = self.__get_line(from_x, from_y, to_x, to_y, a, b)

        dis_per_frame = dis / f_count
        dc = dis_per_frame * math.cos(angle)
        ds = dis_per_frame * math.sin(angle)

        i = 0
        while i < f_count:

            if self.__rem // self.frame_time != 0:
                f_count += 1
                self.__rem -= self.frame_time

            x1 = 0
            y1 = 0
            if from_y == to_y:
                y1 = from_y
                x1 = i * dis_per_frame + from_x if from_x < to_x \
                    else (f_count - i - 1) * dis_per_frame + to_x if from_x > to_x else 0
            elif from_x == to_x:
                x1 = from_x
                y1 = i * dis_per_frame + from_y if from_y < to_y \
                    else (f_count - i - 1) * dis_per_frame + to_y if from_y > to_y else 0
            if from_x < to_x:
                if from_y < to_y:
                    x1 = i * dc + from_x
                    y1 = i * ds + from_y
                elif from_y > to_y:
                    x1 = i * dc + from_x
                    y1 = (f_count - i - 1) * ds + to_y
            elif from_x > to_x:
                if from_y < to_y:
                    x1 = -(i * dc) + from_x - a
                    y1 = i * ds + from_y
                elif from_y > to_y:
                    x1 = -((f_count - i - 1) * dc) + from_x - a
                    y1 = (f_count - i - 1) * ds + to_y

            x2 = x1 + a
            y2 = y1 + b

            x1, x2 = self.__fix(x1, x2, width, a)
            y1, y2 = self.__fix(y1, y2, height, b)

            cut = img[int(y1): int(y2), int(x1): int(x2)]
            cut = cv2.resize(cut, (width, height))

            if i == self.__frame_count - 1:
                self.__rem += self.tempo - (self.frame_time * self.__frame_count)

            self.video.write(cut)
            i += 1

        return f_count

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

    @staticmethod
    def __get_line(x1, y1, x2, y2, a, b):
        angle = math.atan((y2 - y1) / (x2 - x1)) if x1 != x2 and y1 != y2 else 0

        if (x1 < x2 and y1 > y2) or (x1 > x2 and y1 < y2):
            angle *= -1

        x2 += -a if x1 < x2 else a if x1 > x2 else 0
        y2 += -b if y1 < y2 else b if y1 > y2 else 0

        dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return dis, angle

    def slide(self, img, f_count):
        height = img.shape[0]
        width = img.shape[1]
        dis = height // f_count * 2

        offset_y = 0
        i = 0
        while i < f_count:
            back = np.full((height, width, 3), 0, dtype=np.uint8)

            if self.__rem // self.frame_time != 0:
                f_count += 1
                self.__rem -= self.frame_time

            if offset_y > height:
                cut1 = img[0: height - (offset_y - height), 0: width // 2]
                cut2 = img[offset_y - height: height, width // 2: width]
                back[height - cut1.shape[0]: height, 0: cut1.shape[1]] = cut1
                back[0: cut2.shape[0], cut1.shape[1]: cut1.shape[1] + cut2.shape[1]] = cut2
            else:
                cut1 = img[height - offset_y: height, 0: width // 2]
                cut2 = img[0: offset_y, width // 2: width]
                back[offset_y - cut1.shape[0]: offset_y, 0: cut1.shape[1]] = cut1
                back[height - offset_y: height - (offset_y - cut2.shape[0]),
                cut1.shape[1]: cut1.shape[1] + cut2.shape[1]] = cut2

            offset_y += dis
            if i == self.__frame_count - 1:
                self.__rem += self.tempo - (self.frame_time * self.__frame_count)

            self.video.write(back)
            i += 1

        return f_count


def apply_effect(effect, width, height, ignore_time, img_list, i, total_count, index):
    start = ignore_time[0]
    end = ignore_time[len(ignore_time) - 1]
    frame_time = effect.frame_time
    e_count = int(effect.tempo / frame_time)
    back = np.full((height, width, 3), 0, dtype=np.uint8)
    img = img_list[index] if index < len(img_list) else back

    if i >= total_count:
        return

    start_count = to_frame_count(e_count, frame_time, start[1])
    if start[0] == 0 and 0 < start[1] - start[0] and i < start_count:
        i += effect.fade_in(back, img, start_count, start_count)
        apply_effect(effect, width, height, ignore_time, img_list, i, total_count, index + 1)
        return

    end1 = to_frame_count(e_count, frame_time, end[0])
    end2 = to_frame_count(e_count, frame_time, end[1])
    end_count = end2 - end1
    if (total_count - end_count) - end1 <= e_count and end1 <= i:
        effect.fade_in(img, back, end_count, end_count)
        return

    r = int(np.random.rand() * 100)

    config = sm.get_config()
    fade_in = config['effect_chance']['fade_in']
    radial_blur = config['effect_chance']['radial_blur'] + fade_in
    slide = config['effect_chance']['slide'] + radial_blur
    line_move = config['effect_chance']['line_move'] + slide
    fc_mag = config['frame_count_mag']

    e_count *= fc_mag
    if 0 <= r < fade_in and 0 < index < len(img_list):
        i += effect.fade_in(img_list[index - 1], img, e_count // 2, e_count)
        e_type = 'fade_in'
    elif fade_in <= r < radial_blur and 0 < index < len(img_list):
        i += effect.radial_blur(img_list[index - 1], img, 15, e_count, 0.5, 20, 1.3)
        e_type = 'radial_blur'
    elif radial_blur <= r < slide:
        i += effect.slide(img, e_count * 4)
        e_type = 'slide'
    elif slide <= r < line_move:
        pos_x = [0, img.shape[1]]
        pos_y = [0, img.shape[0], img.shape[0] // 2]

        from_pos = [0] * 2
        from_pos[0] = pos_x[int(np.random.rand() * len(pos_x))]
        from_pos[1] = pos_y[int(np.random.rand() * len(pos_y))]

        to_pos = [0] * 2
        to_pos[0] = pos_x[0 if from_pos[0] == pos_x[1] else 1]
        to_pos[1] = pos_y[2 if from_pos[1] == pos_y[2] else 0 if from_pos[1] == pos_y[1] else 1]

        i += effect.line_move(img, e_count, 0.75, from_pos, to_pos)
        e_type = 'line_move'
    else:
        i += effect.normal(img, e_count)
        e_type = 'normal'

    print(f'Effect: {e_type}')
    apply_effect(effect, width, height, ignore_time, img_list, i, total_count, index + 1)


def to_frame_count(count, frame_time, time):
    return count * ((time / frame_time) // count)
