import torch

# init prompt
def init_prompt(model, tokenizer, num_prompts=20, prompt_len=20, device='cpu'):
    tokenizer.add_tokens([f'<prompt{i}>' for i in range(num_prompts)])
    model.resize_token_embeddings(len(tokenizer))

    prompt = "".join([f"<prompt{i}>" for i in range(prompt_len)])

    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids
    prompt_ids = prompt_ids.to(device)

    bos_token = torch.LongTensor([[tokenizer.bos_token_id]]).to(device)
    with torch.no_grad():
        prompt_embed = model.get_input_embeddings()(prompt_ids)
        bos_embed = model.get_input_embeddings()(bos_token)  # [B, 1, E]

    return prompt_ids, prompt_embed, bos_embed

def generate_from_prompt(model,
                         prompt_embed,
                         max_new_tokens=20,
                         topk=1,
                         return_logits=False):
    past = None
    next_token_embed = None
    out_tokens = []
    all_logits = []
    for i in range(max_new_tokens):
        if past is None:
            output = model(inputs_embeds=prompt_embed)
        else:
            output = model(past_key_values=past, input_ids=next_token.long())
        logits = output.logits  # [B, T, V]
        past = output.past_key_values
        logits_i = logits[:, -1, :]  # [B, V]
        if topk > 1:
            next_token = _topk(logits_i, k=topk)  # [B, 1]
        else:
            next_token = _greedy(logits_i)  # [B, 1]
        out_tokens.append(next_token)
        if return_logits:
            all_logits.append(logits_i)

    # handle return values
    out_tokens = torch.cat(out_tokens, dim=-1)  # [B, T]
    return_values = (out_tokens,)

    if return_logits:
        all_logits = torch.stack(all_logits, dim=1)  # [B, T, V]
        return_values += (all_logits,)

    return return_values
