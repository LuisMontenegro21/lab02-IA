# Se emplea Naïve Bayes con el uso de librerías
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# se importa los datos del entrenamiento, se separa en tipo y mensajes
dataframe = pd.read_csv("entrenamiento.txt", sep='\t', names=['emailType', 'mjs'])

# se le asigna a spam 0 y ham 1
email_type_mapping = {'spam':0, 'ham':1}

# Se mapea los labels de texto a valores numéricos
dataframe['emailType'] = dataframe['emailType'].map(email_type_mapping)

# Se divide el dataset entre entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(dataframe['mjs'], dataframe['emailType'], test_size = 0.2, random_state = 12)

# Crear un pipeline con CounterVectorizer y MultinomialNB para el Laplace Smoothing
pipeline = Pipeline([
    ('vectorizador', CountVectorizer()),
    ('clasificador', MultinomialNB(alpha=1.0))
])

# Entrenar el pipeline 
pipeline.fit(X_train, y_train)

# Hacer las predicciones
predictions = pipeline.predict(X_test)

# Evaluar la exactitud del modelo
accuracy = accuracy_score(y_test, predictions)

# Mostrar exactitud
print(f"Exactitud: {accuracy:.2f}")

