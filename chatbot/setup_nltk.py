import nltk

def main():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # add this line
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK resources downloaded.")

if __name__ == "__main__":
    main()
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('stopwords', quiet=True)