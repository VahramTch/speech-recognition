import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import keras
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split



def get_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    mfccs_np = np.mean(mfccs.T, axis=0)

    return mfccs_np

def get_model(num):

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, activation="relu", input_shape=(40,)))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(num, activation="softmax"))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model


def word_segmentation(scale):
    intervals = librosa.effects.split(scale)
    c = len(intervals)-1
    if c>10:
        print("words more than 10")
        print("The program will not run")
        return 0
    elif c<5:
        print("words less than 5")
        print("The program will not run")
        return 0
    else:
        print("words:",c)
        #scale/intervals !!!
        segs = []
        for i in range(len(intervals)):
            segs.append(scale[intervals[i][0]:intervals[i][1]])

        return segs



def get_filters(file):
    scale, sr = librosa.load(file)

    mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spectrogram,x_axis="time",y_axis="mel",sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()




def main():
   file1=[]

   input1 = r"dataset/8words.wav"
   file1.append(r"dataset/album.wav")
   file1.append(r"dataset/covid.wav")
   file1.append(r"dataset/kid.wav")
   file1.append(r"dataset/level.wav")
   file1.append(r"dataset/lockdown.wav")
   file1.append(r"dataset/bird.wav")
   file1.append(r"dataset/napkin.wav")
   file1.append(r"dataset/number.wav")
   #file1.append(r"dataset/margaret.wav")
   #file1.append(r"dataset/child.wav")
   #file1.append(r"dataset/father.wav")


   get_filters(input1)
   scale8, sr8 = librosa.load(input1)
   labels = []
   data = []
   for i in range(len(file1)):
       file_name = file1[i]
       x = file1[i].split("/")
       final_class_label = x[-1].replace(".wav", "")
       labels.append(final_class_label)
       audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
       features = get_features(audio)
       data.append([features, final_class_label])

   segs = word_segmentation(scale8)
   data2 = []
   if segs != 0:
       for i in range(len(segs)):
           features2 = get_features(segs[i])
           data2.append([features2])





   ef = pd.DataFrame(data, columns=['features', 'labels'])
   x = np.array(ef['features'].tolist())
   y = np.array(ef['labels'].tolist())
   labelencoder = LabelEncoder()
   y = to_categorical(labelencoder.fit_transform(y))
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



   model = get_model(len(labels))
   model.fit(x_train, y_train, epochs=10, verbose=1)

   ef2 = pd.DataFrame(data2, columns=['features'])
   x2 = np.array(ef2['features'].tolist())



   loss, acc =model.evaluate(x_test, y_test, verbose=0)
   print("loss:",loss,"acc:",acc)

   y_pred = model.predict(x2)


   k=-1
   m = []
   maxi= []
   for i in range(len(y_pred)):
       m.append(max(y_pred[i]))
       for j in range(len(y_pred)-1):
           if y_pred[i][j] == m[i]:
               maxi.append(j)

   for i in range(len(maxi)):
       print(labels[maxi[i]])









if __name__ == '__main__':
    main()


