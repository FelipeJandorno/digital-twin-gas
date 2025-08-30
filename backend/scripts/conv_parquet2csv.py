import pandas as pd

def main(path):
    df = pd.read_parquet(path)
    df.to_csv("C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\dataset.csv", index=False)

main(path="C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\gas_operation_data.parquet")