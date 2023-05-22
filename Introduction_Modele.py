import pandas as pd 
import numpy as np 
import tensorflow as tf
import glob

#Trouver tous les chemins vers les fichiers qui finissent par .png
liste = glob.glob('./COVID-19_Radiography_Dataset/*/.png')
print(liste)
#Remplacer les \\ par /
liste = list(map(lambda x : [x, x.split('/')[2]], liste))

#Créer un DataFrame pandas
df = pd.DataFrame(liste, columns = ['filepath', 'nameLabel'])
df['label'] = df['nameLabel'].replace(df.nameLabel.unique(), [*range(len(df.nameLabel.unique()))])
df.head()
