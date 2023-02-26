import torch
import torch.nn.functional as F
from prompt_tune import generate_from_prompt


def knowledge_sent_mle(model, prompt_embed, args, teacher_force=True, prompt_ids=None, knowledge_ids=None, return_gen_tokens=False):
    assert prompt_ids is not None and knowledge_ids is not None, "prompt_ids and knowledge_ids are required"
    batch_size, batch_max_len = knowledge_ids.shape
    batch_embed = prompt_embed.repeat(batch_size, 1, 1)  # [B, T, E]

    if teacher_force:
        with torch.no_grad():
            knowledge_emb_in = model.get_input_embeddings()(knowledge_ids[:, :-1])
        input_emb = torch.cat([batch_embed, knowledge_emb_in], dim=1)
        logits = model(inputs_embeds=input_emb).logits
        gen_tokens = torch.topk(logits, k=1, dim=-1).indices.squeeze(-1)
        targets = torch.cat([prompt_ids[:, 1:].repeat(batch_size, 1), knowledge_ids], dim=1)
    else:
        gen_tokens, logits = generate_from_prompt(model, batch_embed, max_new_tokens=batch_max_len, topk=1, return_logits=True)
        targets = knowledge_ids

    loss = F.cross_entropy(logits.permute(0, 2, 1),
                           targets,
                           ignore_index=model.config.pad_token_id,
                           reduction='mean')

    if return_gen_tokens:
        return loss, gen_tokens
    return loss

def dialog_response_mle(model, prompt_embed, args, teacher_force=True, prompt_ids=None, context_ids=None, ref_ids=None):
    assert prompt_ids is not None and context_ids is not None, "prompt_ids and context_ids are required"
    batch_size, batch_max_len = context_ids.shape
    batch_embed = prompt_embed.repeat(batch_size, 1, 1)  # [B, T, E]

    with torch.no_grad():
        context_emb_in = model.get_input_embeddings()(context_ids)
        input_emb = torch.cat([context_emb_in, batch_embed], dim=1)
    if teacher_force:
        with torch.no_grad():
            ref_emb_in = model.get_input_embeddings()(ref_ids[:, :-1])
            input_emb = torch.cat([ref_emb_in, input_emb], dim=1)
        logits = model(inputs_embeds=input_emb).logits
        targets = torch.cat([context_ids[:, 1:], prompt_ids.repeat(batch_size, 1), ref_ids], dim=1)
    else:
        _, logits = generate_from_prompt(model, input_emb, max_new_tokens=batch_max_len, topk=1, return_logits=True)
        targets = ref_ids

    loss = F.cross_entropy(logits.permute(0, 2, 1),
                           targets,
                           ignore_index=model.config.pad_token_id,
                           reduction='mean')

    return loss
