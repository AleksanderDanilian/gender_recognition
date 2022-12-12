import os
import soundfile as sf
import librosa
import numpy as np
import webrtcvad
from tensorflow.python.keras.models import load_model
from tensorflow.keras import backend as K


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


def cut_signal(file, save_folder='', window=1, s_rate=8000, amplify=1, use_array=True, filter_noise=False):
    """
    Функция для нарезки аудио на равные отрезки.
    :param file: str or np.array, путь к файлу или сам файл, в зав-ти от параметра use_array
    :param save_folder: str, путь к папке для сохранения группы файлов длинной window
    :param window: float, длинна нарезки аудио файлов в секундах
    :param s_rate: int, sampling rate, частота дискретизации
    :param amplify: int, усиление сигнала, если требуется
    :param use_array: использовать массивы как для подачи файлов в функцию, так и для вывода из нее (True для продакшна).
    :param filter_noise: Отфильтровывать ли отрезки с малой мощностью сигнала.
    :return:
    cut_signal_array: list, набор массивов временного сигнала длинной window секунд.
    """
    if not use_array:
        signal, s_rate = librosa.load(file, sr=s_rate, dtype=np.float32)
    else:
        signal = file

    signal = signal * amplify
    frames_in_window = s_rate * window
    windows_in_signal = int(len(signal) / frames_in_window)
    cut_signal_array = []
    for i in range(windows_in_signal):
        if filter_noise:
            filter_ = sum(abs(signal[i * frames_in_window: (i + 1) * frames_in_window]))
            if filter_ > 300:  # сохраняем только шумные отрезки
                if use_array:
                    cut_signal_array.append(signal[i * frames_in_window: (i + 1) * frames_in_window])
                else:
                    save_name = os.path.basename(file)
                    sf.write(os.path.join(save_folder, str(i) + save_name),
                             signal[i * frames_in_window: (i + 1) * frames_in_window], s_rate)
            else:
                continue
        else:
            if use_array:
                cut_signal_array.append(signal[i * frames_in_window: (i + 1) * frames_in_window])
            else:
                save_name = os.path.basename(file)
                sf.write(os.path.join(save_folder, str(i) + save_name),
                         signal[i * frames_in_window: (i + 1) * frames_in_window], s_rate)

    return cut_signal_array


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
        chroma_stft_ = np.mean(librosa.feature.chroma_stft(y=signal, sr=s_rate))  # Частота цветности
        audio_features_rest.append(chroma_stft_)
    if rms:
        rms_ = np.mean(librosa.feature.rms(y=signal))  # Среднеквадратичная амплитуда
        audio_features_rest.append(rms_)
    if spec_cent:
        spec_cent_ = np.mean(librosa.feature.spectral_centroid(y=signal, sr=s_rate))  # Спектральный центроид
        audio_features_rest.append(spec_cent_)
    if spec_bw:
        spec_bw_ = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=s_rate))  # Ширина полосы частот
        audio_features_rest.append(spec_bw_)
    if rolloff:
        rolloff_ = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=s_rate))  # Спектральный спад частоты
        audio_features_rest.append(rolloff_)
    if zcr:
        zcr_ = np.mean(librosa.feature.zero_crossing_rate(signal))  # Пересечения нуля
        audio_features_rest.append(zcr_)

    return audio_features_mfcc, audio_features_rest


class PredictGenderAgeNoise:
    """
    Класс для определения пола по аудио файлу.
    """

    def __init__(self, model_gender_path='model_gender_noise_corpus.h5', model_age_path='model_age_corpus.h5'):
        self.model_gender = load_model(model_gender_path, custom_objects={'f1': f1})
        self.model_age = load_model(model_age_path)

    def load_audio_gender_model(self, model_path):
        self.model_gender = load_model(model_path)

    def load_audio_age_model(self, model_path):
        self.model_age = load_model(model_path)

    def analyze(self, file):
        gender_dict = {0: 'Female', 1: 'Male', 2: 'Noise'}
        age_dict = {0: 'teens', 1: 'twenties', 2: 'thirties', 3: 'fourties', 4: 'fifties', 5: 'sixties'}
        file_cleaned = clean_and_cut_audio(file, 8000, 0.3, 60, 1, 0.02)
        signals_cut_array = cut_signal(file=file_cleaned, amplify=1, use_array=True)
        # print(len(signals_cut_array))
        gender = None
        if len(signals_cut_array) == 0:
            print('Длина аудио сигнала менее длины окна. Не смогли нарезать сигнал на window отрезки.')
            gender = 'Noise'  # вероятно, webrtc не нашел отрезков с речью на аудио. Или слишком маленький сигнал
        else:
            gender_confirmed = False
            while not gender_confirmed:
                gender_prediction_array = []
                gender_prediction_mfcc_array = []
                for signal_cut in signals_cut_array:
                    audio_features_mfcc, _ = get_audio_features(signal_cut, mfcc=True, s_rate=8000, use_array=True)

                    gender_prediction_fragment = self.model_gender.predict(np.expand_dims(audio_features_mfcc, 0))[0]
                    gender_prediction_array.append(gender_prediction_fragment)
                    gender_prediction_mfcc_array.append(audio_features_mfcc)
                    if len(gender_prediction_array) > 8:  # предсказываем пол минимум по 8 window отрезкам времени разговора
                        gender_prediction_array = [item for item in gender_prediction_array if item[2] < 0.5] # удаляем arr с высокой вер. шума
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
                    gender_prediction_array = [item for item in gender_prediction_array if item[2] < 0.5] # удаляем arr с выс шумом
                    if len(gender_prediction_array) > 0:
                        male_female_soft_voting = sum(np.array(gender_prediction_array)) / len(gender_prediction_array)
                        print('gender soft voting', male_female_soft_voting)
                        winner_id = int(np.argmax(male_female_soft_voting))
                        gender = gender_dict[winner_id]
                        gender_confirmed = True
                        break
                    else:
                        gender = 'Noise'
                        gender_confirmed = True
                        break

        if gender != 'Noise':
            gender_prediction_mfcc_array = np.array(gender_prediction_mfcc_array)
            if len(gender_prediction_mfcc_array) > 1:
                print(np.array(gender_prediction_mfcc_array).shape)
                gender_prediction_mfcc_array = gender_prediction_mfcc_array[..., np.newaxis]
                age_prediction_array = self.model_age.predict(np.array(gender_prediction_mfcc_array))
                age_soft_voting = sum(np.array(age_prediction_array)) / len(age_prediction_array)
                print('mfcc array', gender_prediction_mfcc_array.shape)
                print('age soft voting', age_soft_voting)
                winner_id = int(np.argmax(age_soft_voting))
                age = age_dict[winner_id]
            else:
                gender_prediction_mfcc_array = gender_prediction_mfcc_array[..., np.newaxis]
                age_prediction = self.model_age.predict(gender_prediction_mfcc_array)[0]
                winner_id = int(np.argmax(age_prediction))
                age = age_dict[winner_id]
        else:
            age = 'Not defined due to Noise factor'

        return gender, age


def get_frames_with_speech(signal, s_rate, greed_level, frame_duration):
    """
    Получаем фреймы длинной frame_duration
    :param signal: array, сигнал для обработки
    :param s_rate: int, sampling rate
    :param greed_level: int, [0,3] - 0 пропускаем много звуков, 3 - пропускаем мало звуков
    :param frame_duration: float, [0.01, 0.02, 0.03] - длина фрейма в секундах
    :return:
    frames_with_speech - array, массив с фреймами, в которых есть "шум"
    """

    vad = webrtcvad.Vad(greed_level)

    audio_length = len(signal) / s_rate
    n_frames = int(audio_length / frame_duration)
    n_signal_cutoffs = int(s_rate * frame_duration)  # number of signal cutoffs to be passed to webrtc (as 1 frame)

    frames_with_speech = []

    for i in range(n_frames):
        sig_to_prcs = np.int16(signal[i * n_signal_cutoffs: (i + 1) * n_signal_cutoffs] * 32768).tobytes()
        is_speech = vad.is_speech(sig_to_prcs, s_rate)
        frames_with_speech.append(is_speech)

    return frames_with_speech


def get_grouped_frames(frames_with_speech, frame_duration, window):
    """
    Группируем фреймы с шумом.
    :param frames_with_speech: arr, фреймы с шумом (или речью)
    :param frame_duration: [0.01, 0.02, 0.03] - длина фрейма в секундах
    :param window: float, окно для нарезки аудио в секундах
    :return:
    grouped_frames - array, сигнал сгруппированный по фреймам со звуком
    n_frames_in_group - int, кол-во фреймов в группе
    """
    n_window_frames = int(len(frames_with_speech) * frame_duration / window)
    grouped_frames = [[] for wdw in range(n_window_frames)]

    n_frames_in_group = int(window / frame_duration)

    for i in range(len(grouped_frames)):
        grouped_frames[i].extend(frames_with_speech[i * n_frames_in_group: (i + 1) * n_frames_in_group])

    return grouped_frames, n_frames_in_group


def get_speech_intervals(grouped_frames, threshold_speech, n_frames_in_group, s_rate, window):
    """
    Получаем интервалы со звуком.
    :param grouped_frames: array, сгруппированный сигнал с фреймами
    :param threshold_speech: [0, 100], порог в процентах. Какой процент фреймов с шумом мы берем, чтобы считать группу
    с речью/шумом.
    :param n_frames_in_group: кол-во фреймов в группе
    :param s_rate: int, семплинг рейт
    :param window: float, окно для нарезки аудио в секундах
    :return:
    speech_intervals - array, инетрвалы сигнала со звуком
    """
    speech_in_group = []

    for group in grouped_frames:
        percentage_speech_true = group.count(True) / n_frames_in_group
        if percentage_speech_true > threshold_speech / 100.0:
            speech_in_group.append(1)
        else:
            speech_in_group.append(0)

    speech_indexes = np.where(np.array(speech_in_group) == 1)[0]

    speech_intervals = []

    for i, idx in enumerate(speech_indexes):
        speech_intervals.append([int(idx * s_rate * window)])
        speech_intervals[i].extend([int((idx + 1) * s_rate * window)])

    return speech_intervals


def get_compound_speech_signal(speech_intervals, signal):
    """
    Из массива массивов получаем 1 сигнал.
    :param speech_intervals: array, инетрвалы сигнала со звуком
    :param signal: array, изначальный сигнал
    :return:
    compound_signal, array - собранный сигнал из кусочков, в которых мы считаем что есть звук/шум.
    """
    compound_signal = []
    for (start, end) in speech_intervals:
        compound_signal.extend(signal[start:end])

    return compound_signal


def get_speech_fragments(signal, s_rate, window, threshold_speech, greed_level, frame_duration):
    """
    Получаем фрагменты с речью/шумом.
    :param signal: array, изначальный аудио сигнал
    :param s_rate: int, семплинг рейт
    :param window: float, окно для нарезки аудио в секундах
    :param threshold_speech: [0, 100], порог в процентах. Какой процент фреймов с шумом мы берем, чтобы считать группу
    :param greed_level: int, [0,3] - 0 пропускаем много звуков, 3 - пропускаем мало звуков
    :param frame_duration: [0.01, 0.02, 0.03] - длина фрейма в секундах
    :return:
    grouped_frames, array - сигнал, сгруппированный по фреймам
    speech_intervals, array - интервалы, в которых есть речь/шум.
    """
    frames_with_speech = get_frames_with_speech(signal, s_rate, greed_level, frame_duration)

    grouped_frames, n_frames_in_group = get_grouped_frames(frames_with_speech, frame_duration, window)

    speech_intervals = get_speech_intervals(grouped_frames, threshold_speech, n_frames_in_group, s_rate, window)

    return grouped_frames, speech_intervals


def clean_and_cut_audio(signal, s_rate, window, threshold_speech, greed_level, frame_duration):
    """
    Проверяем сигнал на наличие отрезков аудио с повышенной энергией используя webrtcvad. Нарезаем сигнал, выбирая только
    отрезки с повышенной энергией.
    :param signal: array, изначальный аудио сигнал
    :param s_rate: int, семплинг рейт
    :param window: float, окно для нарезки аудио в секундах
    :param threshold_speech: [0, 100], порог в процентах. Какой процент фреймов с шумом мы берем, чтобы считать группу
    :param greed_level: int, [0,3] - 0 пропускаем много звуков, 3 - пропускаем мало звуков
    :param frame_duration: [0.01, 0.02, 0.03] - длина фрейма в секундах
    :return:
    compound_signal, array - собранный сигнал из кусочков, в которых мы считаем что есть звук/шум.
    """

    grouped_frames, speech_intervals = get_speech_fragments(signal, s_rate, window, threshold_speech, greed_level,
                                                            frame_duration)

    compound_signal = get_compound_speech_signal(speech_intervals, signal)

    return np.array(compound_signal)
