import utils
import librosa

#  создаем обьект класса PredictGenderNoise
audio_model = utils.PredictGenderNoise('gender_noise_model.h5')

# устанавливаем путь к файлу
file_path = 'test_audio/noise_2.wav'

# получаем сигнал
file, _ = librosa.load(file_path, sr=8000)

# подаем сигнал в метод analyze нашего экземпляра класса
predict = audio_model.analyze(file)

print(predict)

