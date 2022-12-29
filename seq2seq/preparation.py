import json
from sklearn.model_selection import train_test_split
from seq2seq.constants import BATCH_SIZE

def _readFile(file_path: str):
    data = list()
    with open(file_path, "r", encoding="utf-8") as file_data:
        for line in file_data:
            result = json.loads(line.strip())
            if len(result["responses"]) == 0:
                continue
            data.append([result["question"], result["responses"][0]])
    return data

def _splitData(data: list):
    train_set, test_set = train_test_split(data, test_size=0.1, shuffle=True, random_state=42)
    return train_set, test_set

def prepare(file_path):
    data = _readFile(file_path)
    return _splitData(data)