import pandas as pd 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential, load_model


#Définition des repertoires des images et leur target associés
train_dir = 'D:/documents/Documentos Joao/FORMATION DATATEST/PROJET COVID/archive/COVID-19_Radiography_Dataset'
normal_imgs = [fn for fn in os.listdir(f'{train_dir}/Normal/images') if fn.endswith('.png')]
covid_imgs = [fn for fn in os.listdir(f'{train_dir}/COVID/images') if fn.endswith('.png')]
pneumonia_imgs = [fn for fn in os.listdir(f'{train_dir}/Viral Pneumonia/images') if fn.endswith('.png')]
lung_opacity_imgs = [fn for fn in os.listdir(f'{train_dir}/Lung_Opacity/images') if fn.endswith('.png')]

liste = []

for fn in normal_imgs :
    liste.append(f'{train_dir}/Normal/images/' + fn)
for fn in covid_imgs :
    liste.append(f'{train_dir}/COVID/images/' + fn)
for fn in pneumonia_imgs :
    liste.append(f'{train_dir}/Viral Pneumonia/images/' + fn)
for fn in lung_opacity_imgs :
    liste.append(f'{train_dir}/Lung_Opacity/images/' + fn)

liste = list(map(lambda x : [x, x.split('/')[7]], liste))

#Créer un DataFrame pandas
df = pd.DataFrame(liste, columns = ['filepath', 'nameLabel'])
df['label'] = df['nameLabel'].replace(df.nameLabel.unique(), [*range(len(df.nameLabel.unique()))])
df.head(10)


#Charger Exemple Image 
filepath = df.filepath[5]

im = tf.io.read_file(filepath)
im = tf.image.decode_png(im, channels = 3)
plt.imshow(im)


#Charger Jeu de Données & Créer DataSets
def load_image(filepath, resize = (256,256)) :
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels = 1)
    return tf.image.resize(im, resize)

X_train_path, X_test_path, y_train, y_test = train_test_split(df.filepath, df.label, train_size = 0.8, random_state = 123)

dataset_train = tf.data.Dataset.from_tensor_slices((X_train_path, y_train))
dataset_train = dataset_train.map(lambda x,y : [load_image(x), y], num_parallel_calls = -1).batch(32)

dataset_test = tf.data.Dataset.from_tensor_slices((X_test_path, y_test))
dataset_test = dataset_test.map(lambda x,y : [load_image(x), y], num_parallel_calls= - 1).batch(32)

X_test = []

for filepath in tqdm(X_test_path) :
    im = tf.io.read_file(filepath)
    im = tf.image.decode_png(im, channels = 1)
    im = tf.image.resize(im, resize = (256,256))
    X_test.append([im])

X_test = tf.concat(X_test, axis = 0)


#Construire Architecture LeNet

model = Sequential()
model.add(Conv2D(filters = 30, kernel_size = (5,5), padding = 'valid', input_shape = (256,256,1), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'valid', input_shape = (256,256,1), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dropout(rate = 0.2))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))

model.summary()

model.compile('adam', 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Entrainement du modèle
training_history = model.fit(dataset_train, epochs = 5, validation_data = dataset_test)

train_acc_lenet = training_history.history['accuracy']
val_acc_lenet = training_history.history['val_accuracy']

#Evaluation du modèle
y_prob = model.prediction(dataset_test, batch_size = 64)
y_pred = tf.argmax(y_prob, axis = -1).numpy()

print('Accuracy :', accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)


#Prédiction du modèle

indices_random = tf.random.uniform([3], 0, len(X_test), dtype = tf.int32)

plt.figure(figsize = (15,7))

for i, idx in enumerate(indices_random) :
    plt.subplot(1,3,i+1)
    plt.imshow(tf.cast(X_test[idx], tf.int32))
    plt.xticks[]
    plt.yticks[]
    plt.title('Pred class : {} \n Real Class : {}'.format(df.nameLabel.unique[y_pred[idx]], df.nameLabel.unique()[y_test.values[idx]]))