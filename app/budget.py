"""Per-node context budgets. Token counts come from the serving vLLM instance."""
from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass, replace
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NodeBudget:
    context_tokens: int = 8192
    output_tokens: int = 2048
    min_output_tokens: int = 1024
    safety_tokens: int = 256
    retry_reserve_tokens: int = 512

    def __post_init__(self):
        if not (0 < self.min_output_tokens <= self.output_tokens):
            raise ValueError('Require 0 < min_output_tokens <= output_tokens')
        if min(self.safety_tokens, self.retry_reserve_tokens) < 0:
            raise ValueError('Reserves must be nonnegative')
        if self.input_limit <= 0:
            raise ValueError('Budget leaves no room for input')

    @property
    def input_limit(self):
        return self.context_tokens - self.output_tokens - self.safety_tokens - self.retry_reserve_tokens

# Tune these centrally, or pass budget=NodeBudget(...) to validated/pack_records.
NODE_BUDGETS = {
    'default': NodeBudget(),
    'reflect': NodeBudget(output_tokens=1536, min_output_tokens=1024),
    'extract_claims': NodeBudget(output_tokens=2048, min_output_tokens=1536),
    'summarize_one_doc': NodeBudget(output_tokens=2048, min_output_tokens=1024),
    'plan_report': NodeBudget(output_tokens=1280, min_output_tokens=768),
    'write_section': NodeBudget(output_tokens=2048, min_output_tokens=1024, retry_reserve_tokens=0),
    'intro_conclusion': NodeBudget(output_tokens=1536, min_output_tokens=1024),
}

class ContextBudgetError(ValueError):
    pass

class Budgeter:
    def __init__(self, base_url, model):
        root = base_url.rstrip('/')
        if root.endswith('/v1'):
            root = root[:-3]
        self.url = root + '/tokenize'
        self.model = model

    def policy(self, node='default', budget=None):
        p = budget or NODE_BUDGETS.get(node, NODE_BUDGETS['default'])
        # Must match --max-model-len on your server.
        return replace(p, context_tokens=int(os.getenv('MAX_MODEL_LEN', str(p.context_tokens))))

    async def count(self, messages, *, thinking=False):
        payload = {'model': self.model, 'messages': messages,
                   'add_generation_prompt': True,
                   'chat_template_kwargs': {'enable_thinking': thinking}}
        def request():
            req = Request(self.url, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json',
                                   'Authorization': 'Bearer ' + os.getenv('VLLM_API_KEY', 'not-needed')})
            with urlopen(req, timeout=30) as response:
                data = json.load(response)
            count = data.get('count')
            if not isinstance(count, int) or count < 0:
                raise ContextBudgetError('vLLM /tokenize did not return a valid count')
            return count
        # Fail explicitly if counting is unavailable; do not silently estimate.
        return await asyncio.to_thread(request)

    async def allowance(self, messages, *, node='default', budget=None, thinking=False):
        p = self.policy(node, budget)
        count = await self.count(messages, thinking=thinking)
        available = p.context_tokens - count - p.safety_tokens
        if available < p.min_output_tokens:
            raise ContextBudgetError(
                f'{node}: input={count}, available_output={available}, '
                f'minimum_output={p.min_output_tokens}. Batch or compact input first.'
            )
        output = min(p.output_tokens, available)
        logger.info('budget node=%s input=%d output=%d context=%d',
                    node, count, output, p.context_tokens)
        return count, output

    async def pack_records(self, records, build_messages, *, node='default',
                           budget=None, thinking=False):
        """Largest fitting prefix + untouched remainder; never split JSON records.

        build_messages(records) MUST include the schema, just like the real request.
        Caller handles remainder: batch it (claims) or disclose omission (reflection).
        """
        p = self.policy(node, budget)
        if await self.count(build_messages([]), thinking=thinking) > p.input_limit:
            raise ContextBudgetError(f'{node}: fixed prompt/schema already exceeds input budget')
        if not records:
            return [], []
        # Binary search for a fitting prefix; every accepted candidate is counted.
        lo, hi = 0, len(records)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            n = await self.count(build_messages(records[:mid]), thinking=thinking)
            if n <= p.input_limit:
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            raise ContextBudgetError(f'{node}: a single record exceeds the input budget; shorten or chunk it')
        return records[:lo], records[lo:]