import spacy


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
            parse_list.append((token.text, token.dep_, token.head))
        return parse_list

    def get_SVO(self, sentence: str) -> list:
        """
        Estrutura uma sentença em Sujeito-Verbo-Objeto.

        :param sentence: Sentença qualquer em português
        :return: Lista de tuplas no formato ('subject', 'verb', 'object')
        """
        lhs = []
        rhs = []
        root = None
        doc = self.nlp(sentence)
        # nchunks = []
        # nheads = []     
        # for token in doc:
        #         if token.dep_ in ['nsubj', 'obj', 'iobj', 'obl']:
        #             nheads.append(token)
        # for token in nheads:
        #     nchunk = []
        #     for word in token.subtree:
        #         if word.pos_ in ['NOUN', 'VERB']:
        #             nchunk.append(word)
        #     nchunks.append(nchunk)
        # return nchunks

        for token in doc:
            if token.dep_ == 'ROOT':
                root = token.text
            elif root == None:
                lhs.append(token.text)
            else:
                rhs.append(token.text)

        return (' '.join(lhs), root, ' '.join(rhs))
