import os
import logging
from typing import Type, TypeVar
from langfuse import get_client
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from .validate import ainvoke_validated, _with_schema
from .budget import Budgeter, NodeBudget

VLLM_URL = os.environ["VLLM_URL"]
VLLM_MODEL = os.environ["VLLM_MODEL"]

T = TypeVar("T", bound=BaseModel) #input can be any basemodel must be.
logger = logging.getLogger(__name__)


class LLMClients: 
    def __init__(self):
        """Plain chat LLM for prose generation (section writing, stitching)."""
        self.budgeter = Budgeter(VLLM_URL, VLLM_MODEL)
        self._fast = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_URL}/v1",
            api_key="not-needed",
            temperature=0.3,
            max_tokens=4096,
            # Disable Qwen3's <think> tags for structured nodes — they confuse JSON parsing.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self._cheap = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_URL}/v1",
            api_key="not-needed",
            temperature=0.2,
            max_tokens=2048,
            timeout=100,
            # Disable Qwen3's <think> tags for structured nodes — they confuse JSON parsing.
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
                },
        )
        self._reasoner = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_URL}/v1",
            api_key="not-needed",
            temperature=0.3,
            max_tokens=4096,
            timeout=120,
            max_retries=1,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        """For section writers"""
        self._writer = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_URL}/v1",
            api_key="not-needed",
            temperature=0.5,
            max_tokens=2048, 
            extra_body={
                "repetition_penalty": 1.1,
                "chat_template_kwargs": {"enable_thinking": False}},
        )

    # guided decoding path (per-schema FSM)
    def structured_llm(self, schema: Type[T]) -> BaseChatModel:
        return self._fast.with_structured_output(schema, method="json_schema")
    
    def struct_cheap_llm(self, schema: Type[T]) -> BaseChatModel:
        return self._cheap.with_structured_output(schema, method="json_schema")
    
    #prose writer no constraints
    def writer_llm(self, **kwargs):
        return self._writer.bind(**kwargs)
    
    # validate and retry path (free gen + pydantic + error feedback) 
    async def validated(
            self,
            schema: Type[T],
            messages: list[dict],
            tier: str = "cheap",    #models cheap, fast, reasoner
            max_attempts: int = 3,
            fallback: bool = True,
            node: str = "default",
            budget: NodeBudget | None = None,
    ) -> T:
        llm = {"cheap": self._cheap, "fast": self._fast, "reasoner": self._reasoner}[tier]
        #Fallback to no-thinking when using guided decoding
        fb_base = self._cheap if tier == "cheap" else self._fast
        guided = fb_base.with_structured_output(schema, method="json_schema") if fallback else None
        stats: dict = {}
        with get_client().start_as_current_span(name=f"validated:{schema.__name__}") as span:
            try:
                return await ainvoke_validated(
                    llm, schema, messages,
                    max_attempts=max_attempts,
                    guided_fallback=guided,
                    stats=stats,
                    budgeter=self.budgeter, node=node, budget=budget,
                    thinking=(tier == "reasoner"), guided_base=fb_base,
                )
            finally:
                span.update(metadata={"tier": tier, "validation": stats})

    async def pack_records(self, schema, records, build_messages, *,
                               node="default", tier="cheap", budget=None):
            """Budget a document batch including the schema injected by validated()."""
            return await self.budgeter.pack_records(
                records, lambda batch: _with_schema(build_messages(batch), schema),
                node=node, budget=budget, thinking=(tier == "reasoner"))

    async def budgeted_invoke(self, llm, messages, *, node="default", budget=None, thinking=False):
        """Plain (non-JSON) call with max_tokens fitted to the remaining context."""
        count, cap = await self.budgeter.allowance(messages, node=node, budget=budget, thinking=thinking)
        return await llm.bind(max_tokens=cap).ainvoke(messages)
    
    def cheap_llm(self) -> BaseChatModel:
        return self._cheap
    
    def fast_llm(self) -> BaseChatModel:
        return self._fast
    
_clients: LLMClients | None = None

def init_clients() -> LLMClients:
    global _clients
    if _clients is None:
        _clients = LLMClients()
    return _clients

def get_clients() -> LLMClients:
    if _clients is None:
        raise RuntimeError("LLM clients not initialized. Call init_clients() first.")
    return _clients