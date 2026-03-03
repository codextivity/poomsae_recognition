import os
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llm.prompt_templates import build_prompt



os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")


def load_llm(
    adapter_dir: str,
    base_model_override: str | None = None,
    local_files_only: bool = False,
):
    adapter_path = Path(adapter_dir)
    adapter_cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_name = base_model_override or adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("adapter_config.json missing base_model_name_or_path")

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_config,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        hint = (
            "Failed to load base LLM weights. "
            "If you are offline, download the base model and pass a local path via "
            "--llm_base_model or update adapter_config.json. "
            "If you see 'client has been closed', try upgrading/downgrading "
            "huggingface_hub and httpx."
        )
        raise RuntimeError(hint) from exc
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def llm_generate_feedback(
    tokenizer,
    model,
    user_prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only newly generated tokens (exclude prompt echo).
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()



llm_tokenizer, llm_model = load_llm(
                adapter_dir=r'D:\All Docs\All Projects\Git Projects\poomsae_project\pipeline\resources\llm\qwen2_5_1_5b\lora\v2026-02-11',
                base_model_override=r'D:\All Docs\All Projects\Git Projects\poomsae_project\pipeline\resources\llm\qwen2_5_1_5b\base\Qwen2.5-1.5B-Instruct' or None
            )
prompt = build_prompt()

llm_text = llm_generate_feedback(
    tokenizer=llm_tokenizer,
    model=llm_model,
    user_prompt=prompt,
    max_new_tokens=80,
    do_sample=False,
)
print(prompt)

print(llm_text)

