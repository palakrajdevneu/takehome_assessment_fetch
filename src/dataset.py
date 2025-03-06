import torch
from torch.utils.data import Dataset, DataLoader

class DummyMultiTaskDataset(Dataset):
    def __init__(self):
        self.data = [
            ("This is a test sentence.", 0, 1),
            ("I love machine learning!", 1, 0),
            ("Can you tell me the weather?", 2, 2),
            ("What a wonderful day!", 1, 0),
            ("I don't like this movie.", 0, 2)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, class_label, sentiment_label = self.data[idx]
        return {
            "sentence": sentence,
            "classification_label": torch.tensor(class_label, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long)
        }

def get_dataloader(batch_size=2):
    dataset = DummyMultiTaskDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
