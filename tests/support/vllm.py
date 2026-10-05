"""Tests for vllm."""


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Completion:
    text = "generated"
    token_ids = [11, 12]
    finish_reason = "length"
    stop_reason = None


class _SpecialCompletion(_Completion):
    token_ids = [11, 99, 12, 2]
    finish_reason = "stop"
    stop_reason = 2


class _Tokenizer:
    eos_token_id = 2
    all_special_ids = [2, 99]


class _RequestOutput:
    def __init__(self, request_id):
        self.request_id = request_id
        self.outputs = [_Completion()]
        self.finished = True
        self.num_cached_tokens = 3


class _Engine:
    def __init__(self):
        self.requests = []
        self.stepped = False

    def add_request(self, request_id, prompt, sampling_params):
        self.requests.append((request_id, prompt, sampling_params))

    def has_unfinished_requests(self):
        return bool(self.requests) and not self.stepped

    def step(self):
        self.stepped = True
        return [_RequestOutput(request_id) for request_id, _p, _s in self.requests]
