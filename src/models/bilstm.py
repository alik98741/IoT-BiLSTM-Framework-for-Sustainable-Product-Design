import torch
import torch.nn as nn

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=1, task='classification'):
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        out_size = hidden_size * 2
        self.head = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_size, 1 if task=='regression' or num_classes==1 else num_classes)
        )
        if task == 'classification' and num_classes == 1:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        out, _ = self.lstm(x)       # (B, T, 2H)
        last = out[:, -1, :]        # (B, 2H)
        logits = self.head(last)
        return logits

    def predict_prob(self, x):
        logits = self.forward(x)
        if self.task == 'classification' and logits.shape[-1] == 1:
            return self.act(logits).squeeze(-1)
        elif self.task == 'classification':
            return torch.softmax(logits, dim=-1)[:, 1]
        else:
            return logits.squeeze(-1)
