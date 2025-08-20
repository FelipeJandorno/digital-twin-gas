class Pipeline:
    def __init__(self, origem, destino, diametro, comprimento, pressao):
        self.origem = origem  # Nó de origem (ex: "Estação A")
        self.destino = destino  # Nó de destino (ex: "Estação B")
        self.diametro = diametro  # em polegadas
        self.comprimento = comprimento  # em km
        self.pressao = pressao  # em bar