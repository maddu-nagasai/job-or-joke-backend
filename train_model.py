import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Download stopwords if not available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # Load stopwords once

# Load dataset
csv_file = "fake_job_postings.csv"
try:
    df = pd.read_csv(csv_file)  # Load CSV
    print("✅ CSV file loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: The file '{csv_file}' was not found!")
    exit()

# Display column names
print("📌 Available Columns:", df.columns)

# Validate required columns
required_columns = {'description', 'fraudulent'}  # ✅ Fix: Use 'fraudulent' instead of 'label'
if not required_columns.issubset(df.columns):
    print(f"❌ Error: Missing required columns! Required: {required_columns}")
    exit()

# Drop rows with missing values in 'description' or 'fraudulent'
df.dropna(subset=['description', 'fraudulent'], inplace=True)

# Text preprocessing function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Tokenize
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Apply text cleaning
print("🔄 Cleaning text...")  
df['text_cleaned'] = df['description'].apply(clean_text)
print("✅ Text cleaning completed!")

# Convert 'fraudulent' column to 'label' (0 = legitimate, 1 = fake)
df.rename(columns={'fraudulent': 'label'}, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text_cleaned'], df['label'], test_size=0.2, random_state=42)

# Create TF-IDF + Logistic Regression pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('model', LogisticRegression())
])

# Train model
print("🚀 Training model...")
pipeline.fit(X_train, y_train)
print("✅ Model training complete!")

# Save trained model
model_filename = "job_detection_model.pkl"
joblib.dump(pipeline, model_filename)
print(f"💾 Model saved as {model_filename}")
