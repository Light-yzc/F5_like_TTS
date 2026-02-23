import torch
from models.F5_like_text_encoder import CharTokenizer
from data.dataset import collate_fn

def test_collate_mask():
    print("Testing Collate Function with String Length Target Masking...")
    
    # 1. Create a dummy tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["hello world", "[SEP]", "this is a test"])
    
    # 2. Mock items similar to what __getitem__ returns
    # Prompt is 11 chars. " [SEP] " is 7 chars. Target is 14 chars. Total 32 chars.
    batch = [
        {
            "prompt_latent": torch.randn(25, 1024),
            "target_latent": torch.randn(50, 1024),
            "prompt_text_mapped": "hello world",
            "target_text_mapped": "this is a test",
            "full_text": "hello world [SEP] this is a test",
            "target_frames": 50,
        },
        {
            "prompt_latent": torch.randn(10, 1024),
            "target_latent": torch.randn(20, 1024),
            "prompt_text_mapped": "hi",
            "target_text_mapped": "a",
            "full_text": "hi [SEP] a",
            "target_frames": 20,
        }
    ]
    
    # 3. Collate
    result = collate_fn(batch, tokenizer=tokenizer)
    
    input_ids = result["input_ids"]
    attention_mask = result["attention_mask"]
    target_text_mask = result["target_text_mask"]
    
    print("\nBatch shapes:")
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"target_text_mask: {target_text_mask.shape}")
    
    for i in range(len(batch)):
        print(f"\n--- Item {i} ---")
        att_len = attention_mask[i].sum().item()
        tgt_len = target_text_mask[i].sum().item()
        
        tgt_ids = input_ids[i][target_text_mask[i] == 1].tolist()
        prompt_mask = (attention_mask[i] == 1) & (target_text_mask[i] == 0)
        prompt_ids = input_ids[i][prompt_mask].tolist()
        
        prompt_text = "".join([tokenizer.id_to_char.get(id, "<UNK>") for id in prompt_ids])
        tgt_text = "".join([tokenizer.id_to_char.get(id, "<UNK>") for id in tgt_ids])
        
        print(f"Original full string: '{batch[i]['full_text']}'")
        print(f"Total tokens: {int(att_len)} | Target tokens (mask=1): {int(tgt_len)}")
        print(f"Extracted Prompt Area (mask=0): '{prompt_text}'")
        print(f"Extracted Target Area (mask=1): '{tgt_text}'")
        
        assert tgt_len == len(batch[i]["target_text_mapped"])
        assert tgt_text == batch[i]["target_text_mapped"]
        print("✅ Validation passed!")

if __name__ == "__main__":
    test_collate_mask()
