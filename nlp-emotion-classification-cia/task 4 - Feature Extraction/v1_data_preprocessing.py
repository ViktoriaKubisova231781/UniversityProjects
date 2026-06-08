
import contractions
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import os
import pandas as pd

# Function for light preprocessing text
def preprocess_text_light(text):
    """
    Preprocesses the text by:
    - Lower casing the text
    - Fixing the contractions
    - Stripping extra spaces

    Args:
        text (str): The text to preprocess

    Returns:
        str: The preprocessed text
    """
    
    # Lower case the text
    text = text.lower()

    # Fix the contractions
    text = contractions.fix(text)

    # Strip extra spaces
    text = text.strip()

    return text

# Function for medium preprocessing text
def preprocess_text_medium(text):
    """
    Preprocesses the text by:
    - Light preprocessing
    - Removing special characters (except emoticons)

    Args:
        text (str): The text to preprocess

    Returns:
        str: The preprocessed text
    """
    
    # Light preprocessing
    text = preprocess_text_light(text)

    # Remove special characters while preserving emoticons and important punctuation
    text = re.sub(r'[^\w\s!?.,#@:;)(><\U0001F300-\U0001F9FF]', '', text)


    return text

# Function for heavy preprocessing text
def preprocess_text_heavy(text):
    """
    Preprocesses the text by:
    - Medium preprocessing
    - Removing stop words
    - Lemmatizing the text
    - Stemming the text

    Args:
        text (str): The text to preprocess

    Returns:
        str: The preprocessed text
    """
    # Medium preprocessing
    text = preprocess_text_medium(text)

    # Tokenize
    tokens_ = word_tokenize(text)

    # Remove stop words
    tokens_ = [i for i in tokens_ if i not in stopwords.words("english")]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens_ = [lemmatizer.lemmatize(i) for i in tokens_]

    # Stemming
    stemmer = PorterStemmer()
    tokens_ = [stemmer.stem(i) for i in tokens_]

    # Convert tokens back to text
    text = ' '.join(tokens_)

    return text


# Main function
if __name__ == "__main__":

    # Load the data
    df = pd.read_csv("./../Data/go_emotions.csv")

    # Column name to preprocess
    column_name = "text"

    # Copy the dataframe
    df_light = df.copy()
    df_medium = df.copy()
    df_heavy = df.copy()

    # Preprocess the text
    df_light[column_name] = df_light[column_name].apply(preprocess_text_light)
    df_medium[column_name] = df_medium[column_name].apply(preprocess_text_medium)
    df_heavy[column_name] = df_heavy[column_name].apply(preprocess_text_heavy)

    # Save the dataframes
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/data_raw.csv", index=False)
    df_light.to_csv("data/data_light_preprocessed.csv", index=False)
    df_medium.to_csv("data/data_medium_preprocessed.csv", index=False)
    df_heavy.to_csv("data/data_heavy_preprocessed.csv", index=False)