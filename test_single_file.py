import utils
import librosa

#  создаем обьект класса PredictGenderAgeNoise
audio_model = utils.PredictGenderAgeNoise(model_gender_path='gender_model_new_vad.h5',
                                          model_age_path='my_class_model_age_corpus.h5')

# устанавливаем путь к файлу
file_path = 'test_audio/fifties_female.wav'

# получаем сигнал
file, _ = librosa.load(file_path, sr=8000)

# подаем сигнал в метод analyze нашего экземпляра класса (wav=False, значит подаем array)
gender, age = audio_model.analyze(file, wav=False)

print(gender, age)

