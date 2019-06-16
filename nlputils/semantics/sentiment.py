import pickle
from nlputils.lexical.normalizer import Normalizer


class Sentiment:
    """
    Faz análise de sentimentos de textos
    """

    def __init__(self):
        """
        Instancia o objeto configurado para tratar língua portuguesa.
        """
        self.classifier = self.__loadvar('data/dump/classifier.pickle')
        self.transformer = self.__loadvar('data/dump/transformer.pickle')
        self.normalizer = Normalizer()

    def __loadvar(self, path: str) -> object:
        """
        Retorna dump de memória salvo em pickle.

        :param path: Caminho do dump salvo
        :return: Objeto reconstruído
        """
        try:
            with open(path, 'rb') as fp:
                return pickle.load(fp)
        except FileNotFoundError:
            raise FileNotFoundError('Missing file "{}"'.format(path))

    def __preprocessing(self, string: str) -> str:
        """ 
        Retorna uma string sem pontuções, stopwords e com letras em caixa baixa.

        :param string: String qualquer em português
        :return: String transformada
        """
        text = self.normalizer.to_lowercase(string)
        text = self.normalizer.remove_ponctuations(text)
        tokens = self.normalizer.tokenize_words(text)
        tokens = self.normalizer.remove_stopwords(tokens)
        return ' '.join(tokens)

    def sentiment_analysis(self, string: str) -> int:
        """
        Retorna o valor do sentimento identificado em um texto.	

        :param string: String qualquer em português
        :return: Valor entre 0 (sentimento mais negativo) e 5 (sentimento mais positivo)
        """
        normalized = self.__preprocessing(string)
        instance = self.transformer.transform([normalized])
        return int(self.classifier.predict(instance)[0])
