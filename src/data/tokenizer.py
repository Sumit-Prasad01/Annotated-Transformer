import os
import spacy
from utils.logger import logger
from utils.custom_exception import CustomException


def load_tokenizers():
    """
    Load SpaCy tokenizer models for German (de) and English (en).
    Downloads models automatically if they are not already installed.
    """
    try:
        try:
            spacy_de = spacy.load("de_core_news_sm")
        except IOError:
            logger.info("Downloading spacy model de_core_news_sm...")
            os.system("python -m spacy download de_core_news_sm")
            spacy_de = spacy.load("de_core_news_sm")

        try:
            spacy_en = spacy.load("en_core_web_sm")
        except IOError:
            logger.info("Downloading spacy model en_core_web_sm...")
            os.system("python -m spacy download en_core_web_sm")
            spacy_en = spacy.load("en_core_web_sm")

        logger.info("SpaCy tokenizers loaded successfully.")
        return spacy_de, spacy_en
    except Exception as e:
        logger.error("Error loading SpaCy tokenizers.")
        raise CustomException("Failed to load SpaCy tokenizers", e)


def tokenize(text: str, tokenizer) -> list:
    """
    Tokenize a given string using a SpaCy tokenizer instance.
    """
    try:
        return [token.text for token in tokenizer.tokenizer(text)]
    except Exception as e:
        logger.error("Error tokenizing text.")
        raise CustomException("Failed to tokenize text", e)
