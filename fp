import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Pastikan Anda telah mengunduh dataset email untuk training dari NLTK
nltk.download('movie_reviews')

# Membaca dataset dari NLTK
positive_emails = nltk.corpus.movie_reviews.fileids('pos')
negative_emails = nltk.corpus.movie_reviews.fileids('neg')

# Fungsi untuk memproses email dan mengembalikan kata-kata yang telah diproses
def process_email(email):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = word_tokenize(email)
    filtered_words = [ps.stem(w.lower()) for w in words if w.isalpha() and w.lower() not in stop_words]
    return ' '.join(filtered_words)

# Memproses email dan membuat dataset dengan label
dataset = []
for email_id in positive_emails:
    email = nltk.corpus.movie_reviews.raw(email_id)
    processed_email = process_email(email)
    dataset.append((processed_email, 'positive'))

for email_id in negative_emails:
    email = nltk.corpus.movie_reviews.raw(email_id)
    processed_email = process_email(email)
    dataset.append((processed_email, 'negative'))

# Acak dataset
random.shuffle(dataset)

# Membagi dataset menjadi data latih dan data uji
emails, labels = zip(*dataset)
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Membuat vektor fitur dari teks email menggunakan CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Melatih model Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Melakukan prediksi pada data uji
y_pred = classifier.predict(X_test_vectorized)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi: {:.2f}%".format(accuracy * 100))
