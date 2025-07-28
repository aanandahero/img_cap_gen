import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Expanded dataset with more samples per language
data = {
    'text': [
        # English
        'Hello, how are you?', 'Good morning', 'What is your name?',
        'I love programming.', 'The weather is nice today.',
        # French
        'Bonjour, comment ça va?', 'Bonne nuit', 'Quel est ton nom?',
        "J'adore programmer.", "Il fait beau aujourd'hui.",
        # Spanish
        'Hola, ¿cómo estás?', 'Buenos días', '¿Cuál es tu nombre?',
        'Me encanta programar.', 'El clima está agradable hoy.',
        # German
        'Hallo, wie geht es dir?', 'Guten Morgen', 'Wie heißt du?',
        'Ich liebe Programmieren.', 'Das Wetter ist heute schön.',
        # Italian
        'Ciao, come stai?', 'Buongiorno', 'Qual è il tuo nome?',
        'Adoro programmare.', 'Il tempo è bello oggi.',
        # Portuguese
        'Olá, como vai você?', 'Bom dia', 'Qual é o seu nome?',
        'Eu amo programar.', 'O tempo está agradável hoje.'
    ],
    'language': [
        # English
        'English', 'English', 'English', 'English', 'English',
        # French
        'French', 'French', 'French', 'French', 'French',
        # Spanish
        'Spanish', 'Spanish', 'Spanish', 'Spanish', 'Spanish',
        # German
        'German', 'German', 'German', 'German', 'German',
        # Italian
        'Italian', 'Italian', 'Italian', 'Italian', 'Italian',
        # Portuguese
        'Portuguese', 'Portuguese', 'Portuguese', 'Portuguese', 'Portuguese'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['language'], test_size=0.25, random_state=42, stratify=df['language']
)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate accuracy on test data
accuracy = model.score(X_test_tfidf, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

# Interactive prediction loop
print("\nLanguage Detection (type 'exit' to quit)")
while True:
    user_input = input("Enter a sentence: ").strip()
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    if len(user_input) == 0:
        print("Please enter some text.")
        continue
    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)[0]
    print(f"Predicted language: {prediction}")
