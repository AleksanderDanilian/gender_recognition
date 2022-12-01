from tensorflow.python.keras.models import load_model
from tensorflow.keras import backend as K
import collections
import contextlib
import os
import struct
import sys
import wave
import librosa
import numpy as np
import webrtcvad
import soundfile as sf


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_audio_features(file, mfcc=True, chroma_stft=False, rms=False, spec_cent=False, spec_bw=False,
                       rolloff=False, zcr=False, s_rate=8000, use_array=False):
    """
    Функция для сбора фичей с временного сигнала. В дефолтной модели собирался только mfcc.
    :param file: str or nd.array, в зависимости от флага signals_as_array
    :param mfcc: bool, мел-кепстральные коэффициенты. Если True - то собираем эти данные из временного отрезка.
    :param chroma_stft: bool, не использовался в обученной модели.
    :param rms: bool, не использовался в обученной модели.
    :param spec_cent: bool, не использовался в обученной модели.
    :param spec_bw: bool, не использовался в обученной модели.
    :param rolloff: bool, не использовался в обученной модели.
    :param zcr: bool, не использовался в обученной модели.
    :param s_rate: bool, не использовался в обученной модели.
    :param use_array: bool, если True, то на вход подаем массив, а не ссылку на файл
    :return:
    audio_features_mfcc, list - набор массивов, описывающих mfcc аудио файла
    audio_features_rest, list - набор float параметров, описывающих характеристики аудио файла
    """

    audio_features_mfcc = []  # 2d array
    audio_features_rest = []  # 1d array

    if use_array:
        signal = file
    else:
        signal, s_rate = librosa.load(file, sr=s_rate, dtype='float32')

    if mfcc:
        mfcc_ = librosa.feature.mfcc(y=signal, sr=s_rate, n_mfcc=40)  # Мел кепстральные коэффициенты
        audio_features_mfcc.extend(mfcc_)  # .reshape((mfcc_.shape[1], mfcc_.shape[2])))
    if chroma_stft:
        chroma_stft_ = librosa.feature.chroma_stft(y=signal, sr=s_rate)  # Частота цветности
        chroma_stats = [np.mean(chroma_stft_), np.min(chroma_stft_), np.max(chroma_stft_), np.median(chroma_stft_),
                        np.std(chroma_stft_)]
        audio_features_rest.append(chroma_stats)
    if rms:
        rms_ = librosa.feature.rms(y=signal)  # Среднеквадратичная амплитуда
        rms_stats = [np.mean(rms_), np.min(rms_), np.max(rms_), np.median(rms_),
                     np.std(rms_)]
        audio_features_rest.append(rms_stats)
    if spec_cent:
        spec_cent_ = librosa.feature.spectral_centroid(y=signal, sr=s_rate)  # Спектральный центроид
        spec_cent_stats = [np.mean(spec_cent_), np.min(spec_cent_), np.max(spec_cent_), np.median(spec_cent_),
                           np.std(spec_cent_)]
        audio_features_rest.append(spec_cent_stats)
    if spec_bw:
        spec_bw_ = librosa.feature.spectral_bandwidth(y=signal, sr=s_rate)  # Ширина полосы частот
        spec_bw_stats = [np.mean(spec_bw_), np.min(spec_bw_), np.max(spec_bw_), np.median(spec_bw_),
                         np.std(spec_bw_)]
        audio_features_rest.append(spec_bw_stats)
    if rolloff:
        rolloff_ = librosa.feature.spectral_rolloff(y=signal, sr=s_rate)  # Спектральный спад частоты
        rolloff_stats = [np.mean(rolloff_), np.min(rolloff_), np.max(rolloff_), np.median(rolloff_),
                         np.std(rolloff_)]
        audio_features_rest.append(rolloff_stats)
    if zcr:
        zcr_ = librosa.feature.zero_crossing_rate(signal)  # Пересечения нуля
        zcr_stats = [np.mean(zcr_), np.min(zcr_), np.max(zcr_), np.median(zcr_),
                     np.std(zcr_)]
        audio_features_rest.append(zcr_stats)

    return audio_features_mfcc, audio_features_rest


def read_wave(file, wav=True, sr=8000):
    """Reads a .wav file.
    Takes the path(of file in case binary input), and returns (PCM audio data, sample rate).
    """

    if wav:
        print('file_being_sent', file)
        signal, sample_rate = librosa.load(file, sr=8000)
        print('Length of initial signal', len(signal))

        signal = np.int16(signal * 32768)
        tmp = [struct.pack('h', int(val)) for val in signal]
        pcm_data = b''.join(tmp)

        print(len(pcm_data), type(pcm_data))
        return pcm_data, sample_rate
    else:
        return file, sr


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def cut_signals(signal, intervals=1, sr=8000, as_bytes=True):
    """
    Функция нарезки сигнала на куски перед подачей в НС.

    :param signal: arr, сигнал, который мы будем резать на фрагменты
    :param intervals: int or float, время в секундах c которым нарезаем дорожку перед подачей в НС.
    :param sr: int, sampling rate
    :param as_bytes:
    :return:
    cut_signals_list - list, сигнал порезанный на фрагменты
    """
    cut_signals_list = []
    if as_bytes:
        len_window = sr * intervals
        for i in range(int(len(signal) // len_window)):
            cut_signals_list.append(signal[i * len_window:(i + 1) * len_window])

    return cut_signals_list


def get_audio_vad_processed(audio_file, aggressiveness, wav=True, save_files=False, save_folder=''):

    audio, sample_rate = read_wave(audio_file, wav=wav)
    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 100, vad, frames)

    compound_signal = []

    for i, segment in enumerate(segments):
        compound_signal.extend(struct.unpack(int(len(segment) // 2) * 'h', segment))

    compound_signal = np.array(compound_signal) / 32768
    cut_signals_list = cut_signals(compound_signal)

    if save_files:
        for j, sig in enumerate(cut_signals_list):
            file_path = os.path.join(save_folder, str(j) + os.path.basename(audio_file))
            sf.write(file_path, sig, samplerate=sample_rate)

    return cut_signals_list


class PredictGenderNoise:
    """
    Класс для определения пола по аудио файлу.
    """

    def __init__(self, model_path='gender_noise_model.h5'):
        self.model = load_model(model_path, custom_objects={'f1': f1})

    def load_audio_model(self, model_path):
        self.model = load_model(model_path)

    def analyze(self, file, wav):
        gender_dict = {0: 'Female', 1: 'Male', 2: 'Noise'}
        signals_cut_array = get_audio_vad_processed(file, 2, wav=wav, save_files=False, save_folder='')
        gender = None
        if len(signals_cut_array) == 0:
            print('Длина аудио сигнала менее длины окна. Не смогли нарезать сигнал на window отрезки.')
            gender = 'Noise'  # вероятно, webrtc не нашел отрезков с речью на аудио. Или слишком маленький сигнал
        else:
            gender_confirmed = False
            while not gender_confirmed:
                gender_prediction_array = []
                for signal_cut in signals_cut_array:
                    audio_features_mfcc, _ = get_audio_features(signal_cut, mfcc=True, s_rate=8000, use_array=True)

                    gender_prediction_fragment = self.model.predict(np.expand_dims(audio_features_mfcc, 0))[0]
                    gender_prediction_array.append(gender_prediction_fragment)
                    if len(gender_prediction_array) > 8:  # предсказываем пол минимум по 8 window отрезкам времени разговора
                        for i, pred in enumerate(
                                gender_prediction_array):  # если есть window c высокой вероятностью отстутсвия речи - удаляем
                            if pred[2] > 0.5:
                                del gender_prediction_array[i]
                        if len(gender_prediction_array) > 4:
                            male_female_soft_voting = sum(np.array(gender_prediction_array)) / len(
                                gender_prediction_array)
                            winner_id = int(np.argmax(male_female_soft_voting))
                            if male_female_soft_voting[winner_id] > 0.5:
                                gender = gender_dict[winner_id]
                                gender_confirmed = True
                                break
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue

                if gender is None:  # если аудио отрезок оказалася меньше 8 window
                    print(gender_prediction_array)

                    for i, pred in enumerate(
                            gender_prediction_array):  # если есть window c высокой вероятностью отстутсвия речи - удаляем
                        if pred[2] > 0.5:
                            del gender_prediction_array[i]
                    print('after removal of dominant noise', gender_prediction_array)
                    if len(gender_prediction_array) > 0:
                        male_female_soft_voting = sum(np.array(gender_prediction_array)) / len(gender_prediction_array)
                        print('soft voting', male_female_soft_voting)
                        winner_id = int(np.argmax(male_female_soft_voting))
                        gender = gender_dict[winner_id]
                        gender_confirmed = True
                        break
                    else:
                        gender = 'Noise'
                        gender_confirmed = True
                        break

        return gender
