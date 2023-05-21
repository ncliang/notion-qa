from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
import requests


class RemoteChatGLM(LLM):
    remote_host: str = "http://chatglm.nigelliang.com:8000"

    @property
    def _llm_type(self) -> str:
        return "RemoteChatGLM"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        request = {
            "prompt": prompt,
            "history": [],
        }
        resp = requests.post(self.remote_host, json=request)
        assert resp.ok
        return resp.json()["response"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"remote_host": self.remote_host}