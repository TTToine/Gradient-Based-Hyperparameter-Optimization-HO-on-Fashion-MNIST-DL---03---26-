import json
import os
import pandas as pd

def genera_tabella():
    esperimenti = ['baseline', 'truncated', 'reverse', 'hyper_cleaning']
    dati = []

    for exp in esperimenti:
        path = f"risultati_esperimenti/{exp}_metrics.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                metrics = json.load(f)
                
                # Gestiamo la differenza di nomi nelle chiavi tra baseline e meta-learning
                tempo = metrics.get("meta_learning_time_minutes", metrics.get("execution_time_minutes", "N/A"))
                memoria = metrics.get("meta_learning_peak_memory_mb", metrics.get("peak_memory_mb", "N/A"))
                
                dati.append({
                    "Esperimento": exp.capitalize(),
                    "Memoria RAM (MB)": memoria,
                    "Tempo (min)": tempo,
                    "Test Loss": metrics.get("test_loss", "N/A"),
                    "Test Accuracy (%)": metrics.get("test_accuracy_percent", "N/A")
                })

    if dati:
        df = pd.DataFrame(dati)
        print("\n📊 Tabella Riassuntiva per la Relazione:\n")
        print(df.to_markdown(index=False))
    else:
        print("Nessun file JSON trovato in 'risultati_esperimenti'.")

if __name__ == "__main__":
    genera_tabella()
