import spacy
import textacy


class Parser:
    """
    Faz análise sintática de sentenças
    """

    def __init__(self):
        """
        Instancia o objeto configurado para tratar língua portuguesa.
        """
        self.nlp = spacy.load('pt_core_news_sm')

    def parse(self, sentence: str) -> list:
        """
        Faz parsing de uma sentença em português.

        :param sentence: Sentença qualquer em português
        :return: Lista de tuplas no formato ('token', 'syntax_role', 'head')
        """
        parse_list = []
        doc = self.nlp(sentence)
        for token in doc:
            parse_list.append((token.text, token.dep_, token.head.text))
        return parse_list

    def get_SVO(self, sentence: str) -> list:
        """
        Estrutura uma sentença em Sujeito-Verbo-Objeto.

        :param sentence: Sentença qualquer em português
        :return: Lista de tuplas no formato ('subject', 'verb', 'object')
        """
        doc = self.nlp(sentence)
        return textacy.extract.subject_verb_object_triples(doc)
