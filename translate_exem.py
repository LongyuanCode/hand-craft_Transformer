import torch
import torch.nn.functional as F

class TransformerInference:
    def __init__(self, encoder, decoder, src_vocab, trg_vocab, max_len):
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def translate(self, src_sentence):
        src_tokens = [self.src_vocab[token] for token in src_sentence.split()]
        src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度
        src_mask = self.make_src_mask(src_tensor)

        with torch.no_grad():
            encoder_output = self.encoder(src_tensor, src_mask)
        
        trg_tokens = [self.trg_vocab["<SOS>"]]
        for i in range(self.max_len):
            trg_tensor = torch.tensor(trg_tokens, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度
            trg_mask = self.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output = self.decoder(trg_tensor, encoder_output, trg_mask, src_mask)

            pred_token = output.argmax(2)[:,-1].item()
            trg_tokens.append(pred_token)

            if pred_token == self.trg_vocab["<EOS>"]:
                break
        
        translated_sentence = [self.trg_vocab_inv[token] for token in trg_tokens[1:-1]]  # 去除 <SOS> 和 <EOS>
        translated_sentence = " ".join(translated_sentence)
        return translated_sentence

    def make_src_mask(self, src_tensor):
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg_tensor):
        trg_mask = torch.triu(torch.ones((1, trg_tensor.size(1), trg_tensor.size(1))), diagonal=1).bool()
        return trg_mask

# 定义一个简单的 Transformer 编码器和解码器（仅作示例，实际中使用训练好的模型）
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(1000, 512)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_layer, num_layers=6)
    
    def forward(self, src, src_mask):
        src_embedding = self.embedding(src)
        encoder_output = self.transformer_encoder(src_embedding, src_mask)
        return encoder_output

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(1000, 512)
        self.transformer_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = torch.nn.TransformerDecoder(self.transformer_layer, num_layers=6)
        self.linear = torch.nn.Linear(512, 1000)
    
    def forward(self, trg, encoder_output, trg_mask, src_mask):
        trg_embedding = self.embedding(trg)
        decoder_output = self.transformer_decoder(trg_embedding, encoder_output, trg_mask, src_mask)
        output = self.linear(decoder_output)
        return output

# 假设我们有一个已训练好的模型
encoder = Encoder()
decoder = Decoder()
src_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "hello": 3, "world": 4}
trg_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "你好": 3, "世界": 4}
max_len = 20

transformer_inference = TransformerInference(encoder, decoder, src_vocab, trg_vocab, max_len)
src_sentence = "hello world"
translated_sentence = transformer_inference.translate(src_sentence)
print("Translated sentence:", translated_sentence)
