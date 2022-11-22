import utils
import librosa

#  создаем обьект класса PredictGenderNoise
audio_model = utils.PredictGenderNoise('gender_noise_model.h5')

# устанавливаем путь к файлу
file_path = 'test_audio/female_1.wav'

# получаем сигнал
file, _ = librosa.load(file_path, sr=8000)
print('data after librosa', file[:10])
print(type(file))
print(type(file[0]))

# подаем сигнал в метод analyze нашего экземпляра класса
predict = audio_model.analyze(file)

print(predict)

