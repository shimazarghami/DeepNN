import torch
import torch.nn as nn
import torch.optim as optim

# ۱. تنظیمات سخت‌افزاری
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ۲. داده‌های آموزشی
data = [
    ("turn on the kitchen light", "INTENT=LIGHT_ON ROOM=KITCHEN"),
    ("switch on the kitchen light", "INTENT=LIGHT_ON ROOM=KITCHEN"),
    ("turn on the bedroom light", "INTENT=LIGHT_ON ROOM=BEDROOM"),
    ("switch on the living room light", "INTENT=LIGHT_ON ROOM=LIVINGROOM"),
    ("turn off the kitchen light", "INTENT=LIGHT_OFF ROOM=KITCHEN"),
    ("turn off the bedroom light", "INTENT=LIGHT_OFF ROOM=BEDROOM"),
    ("set temperature to 20 degrees", "INTENT=SET_TEMP VALUE=20"),
    ("set temperature to 25 degrees", "INTENT=SET_TEMP VALUE=25"),
    ("decrease temperature to 18", "INTENT=SET_TEMP VALUE=18"),
]

# ۳. ساخت واژه‌نامه (Vocab)
def build_vocab(data):
    in_v = {"<PAD>": 0, "<UNK>": 1}
    out_v = {"<PAD>": 0, "<EOS>": 1} # اضافه کردن EOS برای پایان جمله
    for inp, out in data:
        for w in inp.split():
            if w not in in_v: in_v[w] = len(in_v)
        for t in out.split():
            if t not in out_v: out_v[t] = len(out_v)
    return in_v, out_v

in_vocab, out_vocab = build_vocab(data)
rev_out_vocab = {v: k for k, v in out_vocab.items()}

# ۴. تعریف معماری Seq2Seq ساده

class CommandModel(nn.Module):
    def __init__(self, in_size, out_size, hidden_dim):
        super(CommandModel, self).__init__()
        self.embedding = nn.Embedding(in_size, hidden_dim)
        # Encoder: پردازش جمله ورودی
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Decoder: تولید خروجی بر اساس وضعیت نهایی Encoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_size)

    def forward(self, x, target_len):
        embeds = self.embedding(x)
        # خروجی انکودر را می‌گیریم تا حالت مخفی (h, c) را به دیکودر بدهیم
        _, (h, c) = self.encoder(embeds)
        
        # برای سادگی، یک ورودی صفر به دیکودر می‌دهیم به تعداد کلمات خروجی
        decoder_input = torch.zeros(x.size(0), target_len, embeds.size(-1)).to(device)
        out, _ = self.decoder(decoder_input, (h, c))
        return self.fc(out)

# ۵. مقداردهی مدل و بهینه‌ساز
model = CommandModel(len(in_vocab), len(out_vocab), 64).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ۶. حلقه آموزش
model.train()
for epoch in range(300):
    total_loss = 0
    for inp_text, out_text in data:
        x_indices = [in_vocab.get(w, 1) for w in inp_text.split()]
        y_tokens = [out_vocab[t] for t in out_text.split()] + [out_vocab["<EOS>"]]
        
        x = torch.tensor([x_indices]).to(device)
        y = torch.tensor([y_tokens]).to(device)

        optimizer.zero_grad()
        # ارسال طول واقعی هدف برای جلوگیری از تولید کلمات اضافی
        output = model(x, y.size(1)) 
        
        loss = criterion(output.view(-1, len(out_vocab)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# ۷. تابع پیش‌بینی
def predict(sentence):
    model.eval()
    with torch.no_grad():
        words = sentence.split()
        x = torch.tensor([[in_vocab.get(w, 1) for w in words]]).to(device)
        # فرض می‌کنیم طول خروجی حداکثر ۳ کلمه است
        output = model(x, 3) 
        preds = output.argmax(dim=-1).squeeze().cpu().tolist()
        
        result = []
        for p in preds:
            token = rev_out_vocab[p]
            if token == "<EOS>" or token == "<PAD>": break
            result.append(token)
        return " ".join(result)

print(f"Result: {predict('turn off the bedroom light')}")