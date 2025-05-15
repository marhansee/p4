import torch
import torch.nn as nn
from thop import profile

class Seq2SeqBiGRU(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, output_seq_len=20, output_size=2, dropout_prob=0.2):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len
        self.output_size = output_size

        # Bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=True
        )

        # Decoder GRU (unidirectional)
        self.decoder = nn.GRU(
            input_size=output_size,
            hidden_size=hidden_size * 2,  # because encoder is bidirectional, hidden doubled
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob,
            bidirectional=False
        )

        self.fc_out = nn.Linear(hidden_size * 2, output_size)

        # Learned start token for decoder input
        self.start_token = nn.Parameter(torch.zeros(1, 1, output_size))

    def forward(self, x):
        batch_size = x.size(0)

        # Encode input sequence
        _, hidden = self.encoder(x)  # hidden: (num_layers * 2, batch, hidden_size)

        # Reshape hidden to match decoder expected size: (num_layers, batch, hidden_size * 2)
        # by concatenating forward and backward states
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat((hidden[:,0,:,:], hidden[:,1,:,:]), dim=2)  # (num_layers, batch, hidden_size*2)

        decoder_input = self.start_token.expand(batch_size, 1, self.output_size)

        outputs = []
        hidden_dec = hidden

        for _ in range(self.output_seq_len):
            out_dec, hidden_dec = self.decoder(decoder_input, hidden_dec)
            pred = self.fc_out(out_dec)
            outputs.append(pred)
            decoder_input = pred  # or zeros / teacher forcing

        outputs = torch.cat(outputs, dim=1)
        return outputs
    
def main():
    n_features = 10
    sequence_length = 60
    output_sequence_length = 20
    dummy_input = torch.randn(1, sequence_length, n_features)
    model = Seq2SeqBiGRU(n_features=10, hidden_size=64, num_layers=2, output_seq_len=output_sequence_length)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expecting (1, 20, 2)

    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

if __name__ == '__main__':
    main()