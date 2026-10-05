"""A3 and matched vanilla decode loops with shared KVBench token selection."""
import time
import torch
from helpers.backends.CacheBackend.Sampling import (
    SamplingConfig, make_generator, select_next_token,
)
from helpers.backends.Tokenization import TerminalStopIdsFromTokenizer


def resolve_sampling_policy(sampling_config, label="A3"):
    """Translate the shared resolved config into the cache sampler policy."""
    from core.Sampling import IsGreedy

    supported = {
        "mode", "do_sample", "temperature", "top_p", "top_k", "min_p",
        "typical_p", "repetition_penalty", "presence_penalty",
        "frequency_penalty", "seed", "eos_token_id",
    }
    unknown = set(sampling_config) - supported
    if unknown:
        raise ValueError(f"{label} does not support resolved sampling fields: {sorted(unknown)}")
    fields = {key: sampling_config[key] for key in supported - {"mode", "eos_token_id"}
              if key in sampling_config}
    fields["do_sample"] = not IsGreedy(sampling_config)
    # Greedy sampling ignores temperature, while SamplingConfig validates it.
    fields["temperature"] = max(float(fields.get("temperature", 1.0)), 1e-12)
    if fields.get("top_k", 0) < 0:
        fields["top_k"] = 0
    return SamplingConfig(**fields)


def sampling_stop_ids(tokenizer, sampling_config):
    """Use a sampling EOS override while retaining ragkv's BOS stop rule."""
    bos = getattr(tokenizer, "bos_token_id", None)
    return TerminalStopIdsFromTokenizer(
        tokenizer, sampling_config, explicitStopIds=(() if bos is None else (bos,))
    )

def decode_step(args, model, input, config, stop_list, step, *, sampling=None,
                history=None, generator=None):
    
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
    if sampling is None:
        next_token = torch.argmax(logits.log_softmax(dim=-1), dim=-1)
    else:
        token_id = select_next_token(logits, sampling, history or (), generator)
        next_token = torch.tensor([token_id], device=logits.device, dtype=torch.long)

    # A CUDA argmax is asynchronous. Read the token before recording TTFT,
    # so the boundary is the first token being ready rather than enqueued.
    next_token_id = next_token.item()

    if step == 0:
        first_token_time = time.time()
    else:
        first_token_time = 0

    input['input_ids'] = next_token.unsqueeze(0)
    input['past_key_values'] = outputs.past_key_values
    input['position_ids'] = torch.tensor(
        [input['position_ids'][0][-1].item()+1],
        device=input['position_ids'].device,
    ).unsqueeze(0)
        
    stop = True if next_token_id in stop_list else False
    return input, next_token, stop, first_token_time

def vanilla(
    args, model, tokenizer, input, stop_list, max_new_tokens, config,
    *, return_token_ids=False, sampling=None, generator=None,
):
    new_token_list = []
    generated_token_ids = []

    with torch.no_grad():
        start_time = time.time()
        history_enabled = sampling is not None and (
            sampling.repetition_penalty != 1.0 or sampling.presence_penalty != 0.0
            or sampling.frequency_penalty != 0.0
        )
        history = input['input_ids'][0].tolist() if history_enabled else []
        if sampling is not None and generator is None:
            generator = make_generator(sampling, input['input_ids'].device)
        for i in range(max_new_tokens):
            input, next_token, stop, first_token_time = decode_step(args=args, model=model, input=input, config=config, stop_list=stop_list, step=i, sampling=sampling, history=history, generator=generator)

            if i == 0:
                ttft = first_token_time - start_time
            generated_token_ids.append(next_token.item())
            if history_enabled:
                history.append(next_token.item())
            if stop:
                break

            new_token_list.append(next_token.item())

        tpot = (time.time() - start_time) / (i+1)
        output_sequence = tokenizer.decode(new_token_list, skip_special_tokens=True)
        print(output_sequence)
        if return_token_ids:
            return output_sequence, ttft, tpot, generated_token_ids
        return output_sequence, ttft, tpot

def decode(args, model, tokenizer, input, stop_list, max_new_tokens, config, *, return_token_ids=False,
           sampling=None, generator=None, return_cache=False):
    new_token_list = []
    generated_token_ids = []

    with torch.no_grad():
        start_time = time.time()
        history_enabled = sampling is not None and (
            sampling.repetition_penalty != 1.0 or sampling.presence_penalty != 0.0
            or sampling.frequency_penalty != 0.0
        )
        history = input['input_ids'][0].tolist() if history_enabled else []
        if sampling is not None and generator is None:
            generator = make_generator(sampling, input['input_ids'].device)
        for i in range(max_new_tokens):
            input, next_token, stop, first_token_time = decode_step(args=args, model=model, input=input, config=config, stop_list=stop_list, step=i, sampling=sampling, history=history, generator=generator)
            generated_token_ids.append(next_token.item())
            if history_enabled:
                history.append(next_token.item())
            
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
        # The final sampled token has not been fed through the model yet, so
        # its native KV row is absent from past_key_values. Retention needs
        # that row, but ordinary decode keeps the existing one-forward-per-
        # generated-token timing and behavior.
        if return_cache and generated_token_ids and generated_token_ids[-1] not in stop_list:
            if config == {}:
                outputs = model(input_ids=input['input_ids'], use_cache=True,
                                return_dict=True, past_key_values=input['past_key_values'],
                                position_ids=input['position_ids'])
            else:
                outputs = model(input_ids=input['input_ids'], use_cache=True,
                                return_dict=True, past_key_values=input['past_key_values'],
                                extra_config=config, position_ids=input['position_ids'])
            input['past_key_values'] = outputs.past_key_values
        output_sequence = tokenizer.decode(new_token_list, skip_special_tokens=True)
        print(output_sequence)
        if return_cache:
            if return_token_ids:
                return output_sequence, ttft, tpot, generated_token_ids, input['past_key_values']
            return output_sequence, ttft, tpot, input['past_key_values']
        if return_token_ids:
            return output_sequence, ttft, tpot, generated_token_ids
        return output_sequence, ttft, tpot


def recover_pre_rope_keys(keys, rotary_emb, position_ids):
    """Undo static RoPE on runtime cache keys for zero-position chunk storage."""
    rope_type = getattr(rotary_emb, "rope_type", "default")
    if "dynamic" in str(rope_type).lower():
        raise RuntimeError(f"retain_output does not support dynamic RoPE ({rope_type})")
    from helpers.backends.CacheBackend.RoPE import rotate_half

    cos, sin = rotary_emb(keys, position_ids=position_ids)
    cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    denominator = cos.square() + sin.square()
    if not bool((denominator > 1e-12).all()):
        raise RuntimeError("cannot invert degenerate RoPE factors for retained output")
    return ((keys.float() * cos - rotate_half(keys.float()) * sin)
            / denominator).to(keys.dtype)
