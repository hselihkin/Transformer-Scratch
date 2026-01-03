import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    # d_model: dimension of the model
    # vocab_size: size of the vocabulary

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PostitionalEncoding(nn.Module):
    # d_model: dimension of the model
    # seq_len: maximum sentence/sequence length
    # dropout: dropout rate ( prevent overfitting )

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model) to hold the positional encodings
        pe = torch.zeros(self.seq_len, self.d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )

        # Broadcasting: Multiplying these positions by a frequency vector to create sine/cosine embeddings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # some dimensions represent "meaning" and others represent "order" -- WHY?
        # 1. model can find a specific dimension to be an incredible strong predictor for a task, over-reliant on that specific signal.
        # 2. "Co-adaptation" happens when one part of a neural network only works correctly when another specific part is present.
        # 3. prevents the model from "memorizing" the fixed sinusoidal patterns too rigidly
        # No need to train the positional encodings

        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


# Internal Covariate Shift
# Norm + Scale & Shift
class LayerNormalization(nn.Module):
    # d_model: dimension of the model
    # eps: small value to avoid division by zero

    def __init__(self, d_model: int = 512, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * ((x - mean) / (std + self.eps)) + self.beta


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    # sublayer -> MHA or FFN
    # its not return self.norm(x + sublayer(x)) as it could lead to unstable/vanish gradient and needs "learning rate warmup" period
    # warmup -> slowly increase learning rate.
    # in post-LN: the gradients near the output are much larger than those near the input. and thus need warmup
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Fully Connected NN
# Non-Linearity
# Long-Term Memory: weights of the FFN store general knowledge learned during training.
# Dimension Expansion: features are easier to seperate and process
class FeedForwardLayer(nn.Module):
    # d_model: dimension of the model
    # d_ff: dimension of the feedforward layer
    # dropout: dropout rate

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # W1 & B1
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 & B2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttentionLayer(nn.Module):
    # d_model: dimension of the model
    # h: number of attention heads
    # dropout: dropout rate

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # why heads -> Multiple Perspectives, Computational Efficiency
        d_k = query.shape[-1]

        # @ -> matrix multiplication
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, float("-inf"))

        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (batch_size, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (
            attention_scores @ value
        ), attention_scores  # (batch_size, h, seq_len, d_k)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # attention score for visualization
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


# ===============================================================================================================================


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttentionLayer,
        feed_forward: FeedForwardLayer,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: MultiHeadAttentionLayer,
        cross_attention: MultiHeadAttentionLayer,
        feed_forward: FeedForwardLayer,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    # src_mask -> encoder mask
    def forward(self, enc_output, src_mask, x, trg_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, trg_mask)
        )
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, enc_output, enc_output, src_mask)
        )
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, enc_output, src_mask, x, trg_mask):
        for layer in self.layers:
            x = layer(enc_output, src_mask, x, trg_mask)
        return self.norm(x)


class LinearProjection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        trg_embed: InputEmbeddings,
        src_pos: PostitionalEncoding,
        trg_pos: PostitionalEncoding,
        proj_layer: LinearProjection,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, enc_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(enc_output, src_mask, trg, trg_mask)

    def project(self, x):
        return self.proj_layer(x)


def build_transformer(
    src_vocal_size: int,
    trg_vocab_size: int,
    src_seq_len: int,
    trg_seq_len: int,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    N: int = 6,
    dropout: float = 0.1,
) -> Transformer:
    # Embeddings
    src_embed = InputEmbeddings(d_model, src_vocal_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)

    # Positional Encodings (both do the same thing)
    src_pos = PostitionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PostitionalEncoding(d_model, trg_seq_len, dropout)

    # Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttentionLayer(d_model, h, dropout)
        feed_forward = FeedForwardLayer(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttentionLayer(d_model, h, dropout)
        cross_attention = MultiHeadAttentionLayer(d_model, h, dropout)
        feed_forward = FeedForwardLayer(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            self_attention, cross_attention, feed_forward, dropout
        )
        decoder_blocks.append(decoder_block)

    # Encoder & Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection Layer
    proj_layer = LinearProjection(d_model, trg_vocab_size)

    # Transformer
    transformer = Transformer(
        encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, proj_layer
    )

    # Initializing Params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
