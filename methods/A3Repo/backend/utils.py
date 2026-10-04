"""Upstream greedy decoding used by the A^3 and matched vanilla workers."""
import time
import torch

def decode_step(args, model, input, config, stop_list, step):
    
    if config == {}:
        outputs = model(input_ids=input['input_ids'],
                        use_cache=True,
                        return_dict=True,
                        past_key_values=input['past_key_values'],
                        position_ids=input['position_ids'],)
    else:
        
        outputs = model(input_ids=input['input_ids'],
                        use_cache=True,
                        return_dict=True,
                        past_key_values=input['past_key_values'],
                        extra_config=config,
                        position_ids=input['position_ids'],)  
        
    logits = outputs.logits[:,-1,:]
    next_token_logits = logits.log_softmax(dim=-1)
    next_token = torch.argmax(next_token_logits, dim=-1)

    # A CUDA argmax is asynchronous. Read the token before recording TTFT,
    # so the boundary is the first token being ready rather than enqueued.
    next_token_id = next_token.item()

    if step == 0:
        first_token_time = time.time()
    else:
        first_token_time = 0

    input['input_ids'] = next_token.unsqueeze(0)
    input['past_key_values'] = outputs.past_key_values
    input['position_ids'] = torch.tensor([input['position_ids'][0][-1].item()+1]).unsqueeze(0).cuda()
        
    stop = True if next_token_id in stop_list else False
    return input, next_token, stop, first_token_time

def vanilla(
    args, model, tokenizer, input, stop_list, max_new_tokens, config,
    *, return_token_ids=False,
):
    new_token_list = []
    generated_token_ids = []

    with torch.no_grad():
        start_time = time.time()
        for i in range(max_new_tokens):
            input, next_token, stop, first_token_time = decode_step(args=args, model=model, input=input, config=config, stop_list=stop_list, step=i)

            if i == 0:
                ttft = first_token_time - start_time
            generated_token_ids.append(next_token.item())
            if stop:
                break

            new_token_list.append(next_token.item())

        tpot = (time.time() - start_time) / (i+1)
        output_sequence = tokenizer.decode(new_token_list, skip_special_tokens=True)
        print(output_sequence)
        if return_token_ids:
            return output_sequence, ttft, tpot, generated_token_ids
        return output_sequence, ttft, tpot

def decode(args, model, tokenizer, input, stop_list, max_new_tokens, config, *, return_token_ids=False):
    new_token_list = []
    generated_token_ids = []

    with torch.no_grad():
        start_time = time.time()
        for i in range(max_new_tokens):
            input, next_token, stop, first_token_time = decode_step(args=args, model=model, input=input, config=config, stop_list=stop_list, step=i)
            generated_token_ids.append(next_token.item())
            
            if i == 0:
                ttft = first_token_time - start_time
                if config['reuse_config']:
                    config['reuse_config']['check'] = None
                config['other_config']['decode'] = True
            if stop:
                break
            
            if config['reuse_config']:
                config['reuse_config']['cat_kv'] = None # decode阶段可能还需要根据q做选择
            new_token_list.append(next_token.item())

        tpot = (time.time() - start_time) / (i+1)
        output_sequence = tokenizer.decode(new_token_list, skip_special_tokens=True)
        print(output_sequence)
        if return_token_ids:
            return output_sequence, ttft, tpot, generated_token_ids
        return output_sequence, ttft, tpot
