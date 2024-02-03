import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.language import Language
from spacy.tokens import Doc, Token


@Language.component("lowercase")
def lowercase_doc(doc: Doc) -> Doc:
    """
    A method that lowercases the text of the doc and creates a new doc object
    Args:
        doc (Doc): The doc object

    Returns:
        Doc
    """
    # Lowercase
    lowercase_tokens = [token.lower_ for token in doc]
    lowercase_tokens = [token for token in lowercase_tokens if token.strip()]  # Remove empty strings
    if lowercase_tokens:
        doc = Doc(doc.vocab, words=lowercase_tokens)

    return doc


@Language.component("remove_punctuation")
def remove_punctuation_doc(doc: Doc) -> Doc:
    """
    A method that removes punctuation the text of the doc and creates a new doc object
    Args:
        doc (Doc): The doc object

    Returns:
        Doc
    """
    punctuations = {ord(punct): None for punct in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'}
    no_punctuation_tokens = [token.text.translate(punctuations) for token in doc]
    no_punctuation_tokens = [token for token in no_punctuation_tokens if token.strip()]  # Remove empty strings
    if no_punctuation_tokens:
        doc = Doc(doc.vocab, words=no_punctuation_tokens)

    return doc


@Language.component("remove_stopwords")
def remove_stopwords_doc(doc: Doc) -> Doc:
    """
    A method that removes stopwords from the text of the doc and creates a new doc object
    Args:
        doc (Doc): The doc object

    Returns:
        Doc
    """
    no_stopwords_tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
    no_stopwords_tokens = [token for token in no_stopwords_tokens if token.strip()]  # Remove empty strings
    if no_stopwords_tokens:
        doc = Doc(doc.vocab, words=no_stopwords_tokens)

    return doc


@Language.component("lemmatize")
def lemmatize_doc(doc: Doc) -> Doc:
    """
    A method that removes lemmatize text of the doc and creates a new doc object
    Args:
        doc (Doc): The doc object

    Returns:
        Doc
    """
    lemmatized_tokens = [token.lemma_ for token in doc]
    lemmatized_tokens = [token for token in lemmatized_tokens if token.strip()]  # Remove empty strings
    if lemmatized_tokens:
        doc = Doc(doc.vocab, words=lemmatized_tokens)

    return doc

