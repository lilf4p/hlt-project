import torch
import torch.nn as nn
from transformers import BertModel

class BertAttentionClassifier(nn.Module):

    def __init__(self, bert_model_name='nlpaueb/legal-bert-base-uncased'):
        super(BertAttentionClassifier, self).__init__()
        
        
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=1)
        self.relu=nn.ReLU()
        # Linear layer for classification
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        # Tokenize and encode each text chunk using BERT tokenizer
        
        # Extract BERT outputs (last hidden states)
        
        bert_outputs = [self.bert( ids, mask).pooler_output
                            for (ids, mask)  in zip(input_ids, attention_mask)]
    
        # Stack BERT outputs along the sequence dimension
        stacked_outputs = torch.stack(bert_outputs, dim=1)  # shape: (batch_size, num_chunks, hidden_size)

        # Apply attention across all BERT outputs
        attention_output, _ = self.attention(stacked_outputs,  # (num_chunks, batch_size, hidden_size)
                                             stacked_outputs,  # (num_chunks, batch_size, hidden_size)
                                             stacked_outputs)  # (num_chunks, batch_size, hidden_size)
        attention_output = self.relu(attention_output)
        # Average pooling over the sequence dimension (num_chunks)
        pooled_output = attention_output.mean(dim=0)  # (batch_size, max_length, hidden_size)

        # Apply linear layer for classification
        logits = self.fc(pooled_output)  # (batch_size, 1)
        
        # Squeeze logits to remove extra dimension
        logits = logits.squeeze(dim=-1)  # (batch_size,)
        
        return logits
