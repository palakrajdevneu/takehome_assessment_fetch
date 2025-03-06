import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import MultiTaskSentenceTransformer

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

def train_model(model, dataloader, optimizer, criterion, device, num_epochs=50):
    print("Starting Training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            sentences = batch['sentence']
            class_labels = batch['classification_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)

            optimizer.zero_grad()

            encoded_input = model.tokenizer(
                sentences, padding=True, truncation=True, max_length=128, return_tensors='pt'
            )
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
            
            outputs = model.forward(encoded_input['input_ids'], encoded_input['attention_mask'])
            loss_classification = criterion(outputs['classification_logits'], class_labels)
            loss_sentiment = criterion(outputs['sentiment_logits'], sentiment_labels)
            total_loss = loss_classification + loss_sentiment

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * len(sentences)
            total_samples += len(sentences)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/total_samples:.4f}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = DummyMultiTaskDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = MultiTaskSentenceTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    train_model(model, dataloader, optimizer, criterion, device)

if __name__ == '__main__':
    main()

