import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_classification_classes=3, num_sentiment_classes=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.transformer.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.classification_head = nn.Linear(self.hidden_size, num_classification_classes)
        self.sentiment_head = nn.Linear(self.hidden_size, num_sentiment_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask

        sentence_embeddings = self.dropout(sentence_embeddings)
        classification_logits = F.softmax(self.classification_head(sentence_embeddings), dim=1)
        sentiment_logits = F.softmax(self.sentiment_head(sentence_embeddings), dim=1)

        return {
            'sentence_embedding': sentence_embeddings,
            'classification_logits': classification_logits,
            'sentiment_logits': sentiment_logits
        }

