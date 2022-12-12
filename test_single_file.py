import utils
import librosa

#  создаем обьект класса PredictGenderNoise
audio_model = utils.PredictGenderAgeNoise('model_gender_noise_corpus.h5', 'model_age_corpus.h5')

# устанавливаем путь к файлу
file_path = 'test_audio/fifties_female.wav'

# получаем сигнал
file, _ = librosa.load(file_path, sr=8000)

# подаем сигнал в метод analyze нашего экземпляра класса
predict = audio_model.analyze(file)

print(predict)

