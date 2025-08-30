import pandas as pd

def main(path):
    df = pd.read_csv(path)
    df.to_parquet("C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\deepseek.parquet", index=False)

main(path="C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\deepseek_csv.txt")