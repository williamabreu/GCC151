class Parser:
    """
    Faz análise sintática de sentenças
    """

    def __init__(self):
        """
        Instancia o objeto configurado para tratar língua portuguesa.
        """
        pass

    def parse(self, sentence: str) -> list:
        """
        Faz parsing de uma sentença em português.

        :param sentence: Sentença qualquer em português
        :return: Lista de tuplas no formato ('token', 'syntax_role', 'head')
        """
        pass

    def get_SVO(self, sentence: str) -> list:
        """
        Estrutura uma sentença em Sujeito-Verbo-Objeto

        :param sentence: Sentença qualquer em português
        :return: Lista de tuplas no formato ('subject', 'verb', 'object')
        """
        pass
