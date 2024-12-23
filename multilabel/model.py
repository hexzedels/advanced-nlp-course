import torch
from torch import nn, Tensor
from transformers import AutoModel

from const import BASE_MODEL_NAME, LABELS

class MultiLabelPoolerWithHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, last_hidden_state: Tensor):
        # last hidden state shape: [BATCH x SEQ_LEN x HIDDEN_SIZE]
        # Seq: [CLS], Tok1, Tok2, Tok3, [SEP]
        x = last_hidden_state[:, 0]  # [BATCH x HIDDEN_SIZE]
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)  # [BATCH x NUM_CLASSES]
        return x
    

class MultiLabelAvgPoolerWithHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(hidden_size, num_classes)

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def forward(
            self,
            last_hidden_state: Tensor,
            attention_mask: Tensor,
            ):
        # last hidden state shape: [BATCH x SEQ_LEN x HIDDEN_SIZE]
        # Seq: [CLS], Tok1, Tok2, Tok3, [SEP]

        x = self.average_pool(last_hidden_state, attention_mask)
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)  # [BATCH x NUM_CLASSES]
        return x
    

class MultiLabelModel(nn.Module):
    def __init__(self, avg: bool):
        super().__init__()
        self.avg = avg

        self.encoder = AutoModel.from_pretrained(BASE_MODEL_NAME)
        if avg == True:
            self.pooler_head = MultiLabelAvgPoolerWithHead(self.encoder.config.hidden_size, len(LABELS)) # average
        else:
            self.pooler_head = MultiLabelPoolerWithHead(self.encoder.config.hidden_size, len(LABELS)) # baseline


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        encoder_outs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = encoder_outs.last_hidden_state
        if self.avg == True:
            logits = self.pooler_head(last_hidden_state, attention_mask) # average
        else:
            logits = self.pooler_head(last_hidden_state) # baseline
        return logits

class MultiLabelWrap(nn.Module):
    def __init__(self, model: MultiLabelModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        probas = torch.sigmoid(logits)
        return probas
