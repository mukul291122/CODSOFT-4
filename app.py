import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
import string
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load the dataset
df = pd.read_csv(r'C:\Mukul\SMS spam\dataset\spam.csv', encoding='latin-1')

# Drop unnecessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Preprocess text data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(preprocess_text)

# Convert labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_tfidf_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf_res, y_train_res)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

# Tkinter GUI
def detect_spam():
    input_message = message_entry.get()
    processed_message = preprocess_text(input_message)
    message_tfidf = vectorizer.transform([processed_message])
    prediction = model.predict(message_tfidf)
    result = "Spam" if prediction[0] else "Ham"
    messagebox.showinfo("Result", f'The message is: {result}')

# Create the main window
root = tk.Tk()
root.title("Spam Detector")

# Set the window to maximized state
root.state('zoomed')

# Load background image and resize it to fit the window
bg_image_path = r'C:\Mukul\SMS spam\1.jpeg'
bg_image = Image.open(bg_image_path)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
bg_image_tk = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(root, image=bg_image_tk)
background_label.place(relwidth=1, relheight=1)

# Create and place the label, entry, and button with larger font and size
message_label = tk.Label(root, text="Enter a message:", font=("Arial", 20), bg="lightblue")
message_label.place(relx=0.5, rely=0.3, anchor='center')

message_entry = tk.Entry(root, width=50, font=("Arial", 18))
message_entry.place(relx=0.5, rely=0.4, anchor='center')

detect_button = tk.Button(root, text="Detect", command=detect_spam, font=("Arial", 20), bg="lightgreen")
detect_button.place(relx=0.5, rely=0.5, anchor='center')

# Run the application
root.mainloop()
