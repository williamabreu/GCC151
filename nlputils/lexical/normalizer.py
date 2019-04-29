import nltk
import unidecode
import string


class Normalizer:
    """
    Aplica rotinas de normalização de texto
    """

    def __init__(self):
        """
        Instancia o objeto configurado para tratar língua portuguesa.
        """
        self.sent_tokenizer = nltk.data.load(
            'tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()

    def remove_ponctuations(self, text: str) -> str:
        """
        Remove sinais de pontuação em uma string.

        :param text: Texto de entrada
        :return: Texto sem as pontuações
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_accents(self, text: str) -> str:
        """
        Remove acentuações em uma string.

        :param text: Texto de entrada
        :return: Texto sem as acentuações
        """
        return unidecode.unidecode(text)

    def remove_stopwords(self, tokens: list) -> list:
        """
        Remove stopwords de uma lista de palavras.

        :param tokens: Lista de palavras
        :return: A lista sem as stopwords
        """
        stopwords = nltk.corpus.stopwords.words('portuguese')
        return [word for word in tokens if word not in stopwords]

    def to_lowercase(self, text: str) -> str:
        """
        Converte caracteres para caixa baixa.

        :param text: Texto de entrada
        :return: Texto convertido em caixa baixa
        """
        return text.lower()

    def stemmize(self, tokens: list) -> list:
        """
        Converte palavras para seus radicais.

        :param text: Lista de palavras
        :return: Lista com os radiciais das palavras
        """
        return [self.stemmer.stem(word) for word in tokens]

    def tokenize_sentences(self, text: str) -> list:
        """
        Tokeniza texto em sentenças delimitadas por ponto final.

        :param text: Texto de entrada
        :return: Lista de sentenças do texto
        """
        return self.sent_tokenizer.tokenize(text)

    def tokenize_words(self, text: str) -> list:
        """
        Tokeniza texto em palavras.

        :param text: Texto de entrada
        :return: Lista de palavras do texto
        """
        return nltk.tokenize.word_tokenize(text)
