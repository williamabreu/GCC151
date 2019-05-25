import spacy


class Tagger:
    """
    Faz etiquetagem morfossintática
    """

    def __init__(self):
        """
        Instancia o objeto configurado para tratar língua portuguesa.
        """
        self.nlp = spacy.load('pt_core_news_sm')

    def tag(self, string: str) -> list:
        """
        Faz etiquetagem de uma string qualquer.

        :param string: String qualquer em português
        :return: Lista de tuplas no formato ('token', 'morphosyntax_tag')
        """
        tag_list = []
        tagged_text = self.nlp(string)
        for token in tagged_text:
            tag_list.append((token, token.pos_))
        return tag_list
