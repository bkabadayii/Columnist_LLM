import pandas as pd
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
import torch

# bits and bytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Path to your fine-tuned model
model_path = "./mistral_tr_finetuned_hilalkaplan"
base_model = "malhajar/Mistral-7B-Instruct-v0.2-turkish"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input prompt
prompt = "Write an article about the importance of renewable energy."

# Load the tokenizer
print("Loading the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer is successfully loaded\n")

# Load the fine-tuned model
print("Loading the model...")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", use_safetensors=True)
print("Model is successfully loaded\n")

# Input prompt
prompt = """
### Instruction: Aşağıdaki gerçekler verildiğinde:
- Peşmerge Operasyon Birimi Komutanı, TSK'nın peşmergeye verdiği eğitimin Şengal'in DAEŞ'ten kurtarılmasında belirleyici olduğunu açıkladı.
- TSK, Musul'un Başika kasabasındaki askeri üsse askeri takviye gerçekleştirdi.
- Irak Dışişleri Bakanlığı, Türk askerinin Musul'daki varlığı nedeniyle Türkiye'den askerlerini çekmesini talep etti.
- TSK'nın bölgedeki varlığı, uluslararası koalisyon çerçevesinde DAEŞ'e karşı mücadelede destek olarak kabul ediliyor.
- Türkiye'nin Musul'daki askeri varlığı 25 tank ve 400 askerle güçlendirildi.

Ve tarif edilen olay:
Türkiye, Musul'un kontrolündeki Başika kasabasındaki askeri varlığını artırarak, bölgedeki gerilimlere neden oldu. Irak ve İran, bu hareketi egemenliğe aykırı bularak eleştirirken, Türkiye ise askeri varlığını DAEŞ'le mücadelede bir destek olarak gördüğünü belirtmektedir. Bu durum, Irak ve Kürdistan Bölgesel Yönetimi arasındaki güç dinamiklerini etkilemiştir.

Bu olayın daha geniş etkileri hakkında bir köşe yazısı yazın. Olayın kamu politikalarını, toplumsal tutumları veya gelecekteki gelişmeleri nasıl etkileyebileceğini göz önünde bulundurun.

### Response:
"""

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        attention_mask=inputs.attention_mask,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,  # Number of sequences to generate
        do_sample=True,  # If you want to add randomness for creative text
        temperature=0.7,  # Lower value for more focused output, higher for randomness
        repetition_penalty=1.3
    )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("MODEL RESPONSE")
print("--------------")
print(generated_text)