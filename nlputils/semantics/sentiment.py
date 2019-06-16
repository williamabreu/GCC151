import pickle
import os
import sys
from nlputils.lexical.normalizer import Normalizer


class Sentiment:
    """
    Faz análise de sentimentos de textos
    """

    def __init__(self):
        """
        Instancia o objeto configurado para tratar língua portuguesa.
        """
        try:
            with open('data/dump/LR_sentiment', 'rb') as fp:
                self.classifier_lr = pickle.load(fp)
                fp.close()

            with open('data/dump/Transformer', 'rb') as fp:
                self.transformer = pickle.load(fp)
                fp.close()
        except FileNotFoundError:
            print('Você precisa fazer o download dos dumps e colocar na pasta /data/dump/')
            sys.exit(1)

        self.normalizer = Normalizer()


    def preprocessing(self, string: str) -> str:
        """ 
        Retorna uma string sem pontuções, stopwords e com letras em caixa baixa

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
        preprocessed_sentence = self.preprocessing(string)
        instance = self.transformer.transform([self.preprocessing(string)])
        return self.classifier_lr.predict(instance)
