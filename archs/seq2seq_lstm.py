import torch
import torch.nn as nn
from thop import profile

class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, output_seq_len=20, output_size=2, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.output_size = output_size

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=output_size,  # decoder inputs are previous outputs
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Output projection layer
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Learned start token to kick off the decoder
        self.start_token = nn.Parameter(torch.zeros(1, 1, output_size))

    def forward(self, x):
        batch_size = x.size(0)

        # Encode input sequence
        _, (hidden, cell) = self.encoder(x)  # hidden, cell: (num_layers, batch, hidden_size)

        # Prepare decoder inputs: repeat start token for batch
        decoder_input = self.start_token.expand(batch_size, 1, self.output_size)  # (batch, 1, output_size)

        outputs = []
        hidden_dec = hidden
        cell_dec = cell

        # Decode for output_seq_len steps
        for _ in range(self.output_seq_len):
            out_dec, (hidden_dec, cell_dec) = self.decoder(decoder_input, (hidden_dec, cell_dec))
            pred = self.fc_out(out_dec)  # (batch, 1, output_size)
            outputs.append(pred)
            decoder_input = pred 

        # Concatenate all predictions: (batch, output_seq_len, output_size)
        outputs = torch.cat(outputs, dim=1)
        return outputs

def main():
    n_features = 10
    sequence_length = 60
    output_sequence_length = 20
    dummy_input = torch.randn(1, sequence_length, n_features)
    model = Seq2SeqLSTM(n_features, hidden_size=64, num_layers=2, output_seq_len=output_sequence_length)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 20, 2)

    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()