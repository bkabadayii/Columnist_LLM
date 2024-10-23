from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"

system_msg ="Sen bir türk gazetecisin. Görevin kullanıcının istediği bir konu hakkında köşe yazısı yazmak."

prompt = """Aşağıdaki gerçekler verildiğinde:
- Peşmerge Operasyon Birimi Komutanı, TSK'nın peşmergeye verdiği eğitimin Şengal'in DAEŞ'ten kurtarılmasında belirleyici olduğunu açıkladı.
- TSK, Musul'un Başika kasabasındaki askeri üsse askeri takviye gerçekleştirdi.
- Irak Dışişleri Bakanlığı, Türk askerinin Musul'daki varlığı nedeniyle Türkiye'den askerlerini çekmesini talep etti.
- TSK'nın bölgedeki varlığı, uluslararası koalisyon çerçevesinde DAEŞ'e karşı mücadelede destek olarak kabul ediliyor.
- Türkiye'nin Musul'daki askeri varlığı 25 tank ve 400 askerle güçlendirildi.

Ve tarif edilen olay:
Türkiye, Musul'un kontrolündeki Başika kasabasındaki askeri varlığını artırarak, bölgedeki gerilimlere neden oldu. Irak ve İran, bu hareketi egemenliğe aykırı bularak eleştirirken, Türkiye ise askeri varlığını DAEŞ'le mücadelede bir destek olarak gördüğünü belirtmektedir. Bu durum, Irak ve Kürdistan Bölgesel Yönetimi arasındaki güç dinamiklerini etkilemiştir.

Bu olayın daha geniş etkileri hakkında bir köşe yazısı yazın. Olayın kamu politikalarını, toplumsal tutumları veya gelecekteki gelişmeleri nasıl etkileyebileceğini göz önünde bulundurun."""

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": prompt},
]


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))