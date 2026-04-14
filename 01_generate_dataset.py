import os
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY não encontrada no .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

TOTAL = 60
BATCH = 5

SYSTEM_PROMPT = (
    "Você é um gerador de dados de treinamento para fine-tuning de um "
    "assistente especialista em gestão de estoque e WMS (Warehouse Management System). "
    "Gere exatamente {n} pares únicos e realistas no formato JSON, cobrindo temas como: "
    "controle de inventário, curva ABC, acurácia de estoque, picking, packing, putaway, "
    "endereçamento, FIFO/FEFO/LIFO, giro de estoque, ponto de pedido, níveis de serviço, "
    "SKU, cross-docking, recebimento, expedição, indicadores (KPIs) de WMS, "
    "integração ERP/WMS, coletor de dados e RFID. "
    "Retorne APENAS um array JSON válido, sem markdown, sem comentários, no formato: "
    '[{{"prompt": "pergunta do usuário", "response": "resposta técnica e detalhada"}}, ...]'
)


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def generate_batch(n: int):
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": SYSTEM_PROMPT.format(n=n)}]
    )
    data = extract_json(resp.choices[0].message.content)
    return [
        {"prompt": str(d["prompt"]), "response": str(d["response"])}
        for d in data
        if isinstance(d, dict) and "prompt" in d and "response" in d
    ]


def main():
    pairs = []
    with tqdm(total=TOTAL, desc="Gerando pares") as pbar:
        while len(pairs) < TOTAL:
            remaining = TOTAL - len(pairs)
            size = min(BATCH, remaining)
            try:
                batch = generate_batch(size)
            except Exception as e:
                tqdm.write(f"Erro no batch: {e}")
                continue
            batch = batch[:remaining]
            pairs.extend(batch)
            pbar.update(len(batch))

    random.shuffle(pairs)
    split = int(len(pairs) * 0.9)
    train, test = pairs[:split], pairs[split:]

    out_dir = Path("dataset")
    out_dir.mkdir(exist_ok=True)

    for name, rows in (("train", train), ("test", test)):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total de pares gerados: {len(pairs)}")
    print(f"  train: {len(train)} -> dataset/train.jsonl")
    print(f"  test:  {len(test)} -> dataset/test.jsonl")


if __name__ == "__main__":
    main()