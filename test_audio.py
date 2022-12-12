import os
import librosa
import numpy as np
import utils
from sklearn.metrics import f1_score

audio_model = utils.PredictGenderAgeNoise('gender_noise_model.h5')

predict = []
female_audio_counter = 0
male_audio_counter = 0
correct_answers = 0
odds = [str(i) for i in range(1, 24, 2)]
evens = [str(i) for i in range(0, 25, 2)]

base_folder = 'C:\\Users\\ale-d\\PycharmProjects\\gender_audio_recognition\\ravdess\\'
predicts = []
real_values = []

gender_dict = {'Female': 0, 'Male': 1, 'Noise': 2}


for folder in os.listdir(base_folder)[:6]:
    if int(folder[-1]) % 2 == 0:
        print('--------------supposed to be female folder----------------')
        print(folder)
        female_audio_counter += len(os.listdir(base_folder + folder))
        for file in os.listdir(base_folder + folder):
            file_path = os.path.join(base_folder, folder, file)
            file, _ = librosa.load(file_path, sr=8000)
            predict = audio_model.analyze(file)
            if predict == 'Female':
                correct_answers += 1
            else:
                print(predict)
            predicts.extend([gender_dict[predict]])
        real_values.extend(np.zeros(len(os.listdir(base_folder + folder))))

    else:
        print('--------------supposed to be male folder----------------')
        print(folder)
        male_audio_counter += len(os.listdir(base_folder + folder))
        for file in os.listdir(base_folder + folder):
            file_path = os.path.join(base_folder, folder, file)
            file, _ = librosa.load(file_path, sr=8000)
            predict = audio_model.analyze(file)
            if predict == 'Male':
                correct_answers += 1
            else:
                print(predict)
            print(predict)
            predicts.extend([gender_dict[predict]])
        real_values.extend(np.ones(len(os.listdir(base_folder + folder))))


noise_folder = 'C:\\Users\\ale-d\\PycharmProjects\\gender_audio_recognition\\acoustic_scenes\\audio_not_used\\'
noise_audio_counter = 0
nr_of_noise_sounds = 240

for file in os.listdir(noise_folder)[:nr_of_noise_sounds]:
    noise_audio_counter += 1
    file_path = os.path.join(noise_folder, file)
    file, _ = librosa.load(file_path, sr=8000)
    predict = audio_model.analyze(file)
    if predict == 'Noise':
        correct_answers += 1
    else:
        print(predict)
    predicts.extend([gender_dict[predict]])
real_values.extend([2 for i in range(nr_of_noise_sounds)])

print(predicts)
print(real_values)
print(np.array(predicts).shape)
print(np.array(real_values).shape)
print()
percent_of_correct = correct_answers / (male_audio_counter + female_audio_counter + noise_audio_counter)
print('процент верных ответов', percent_of_correct)
print('мужские голоса', male_audio_counter)
print('женские голоса', female_audio_counter)
print('шум', noise_audio_counter)
print('кол-во правильных ответов', correct_answers)

print('f1_micro score', f1_score(predicts, real_values, average='micro'))
