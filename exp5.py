
import spacy
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Medical text for processing
text = """
The patient presented with a history of bronchial asthma, hyperlipidemia, and hypothyroidism. The physician
prescribed Salbutamol, Atorvastatin, and Levothyroxine as part of the treatment plan. A chest X-ray revealed mild
bilateral pulmonary infiltrates, while blood investigations showed elevated cholesterol levels and abnormal thyroid
hormone levels.
"""

# Tokenize the text
tokens = word_tokenize(text)

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_words = [
    word
    for word in tokens
    if word.lower() not in stop_words
]

print("Filtered Words:")
print(filtered_words)

# Process text using spaCy
doc = nlp(text)

# Extract named entities
print("\nNamed Entities:")

for ent in doc.ents:
    print(ent.text, "->", ent.label_)
