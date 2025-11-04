# step_back_pipeline.py
import ollama
from typing import List, Dict, Any

# --- Minimal prompt template (LangChain-like) ---
class PromptTemplate:
    def __init__(self, messages: List[Any]):
        """
        messages: list of either
          - ("role", "content with {vars}")
          - a list/tuple of such items (for few-shot blocks)
        """
        self.messages = messages

    @staticmethod
    def from_messages(messages: List[Any]) -> "PromptTemplate":
        return PromptTemplate(messages)

    def format(self, **kwargs) -> List[Dict[str, str]]:
        def _safe_format(content: str, mapping: Dict[str, Any]) -> str:
            # Replace only explicit placeholders like {question}, {text}, ...
            # without interpreting other braces in the message text (e.g. JSON examples).
            out = content
            for k, v in mapping.items():
                out = out.replace("{" + k + "}", str(v))
            return out

        def _expand(item):
            # item can be ("role", "content") OR a nested list/tuple
            if isinstance(item, (list, tuple)) and len(item) and isinstance(item[0], (list, tuple)):
                # nested block: flatten
                out = []
                for sub in item:
                    out += _expand(sub)
                return out
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                role, content = item
                return [{"role": role, "content": _safe_format(content, kwargs)}]
            else:
                raise ValueError("Invalid message item: %r" % (item,))

        rendered: List[Dict[str, str]] = []
        for m in self.messages:
            rendered += _expand(m)
        return rendered


# --- LLM wrapper (returns a string like StrOutputParser) ---
class ChatOllamaMini:
    def __init__(self, model: str = "gemma3:1b", temperature: float = 0.0, base_url: str = "http://localhost:11434"):
        # Use a client so we can set base_url explicitly
        self.client = ollama.Client(host=base_url)
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return resp["message"]["content"]


# --- Your few-shot block (examples) ---
few_shot_prompt = [
    ("user", "Who is the president of a given country X?"),
    ("assistant", "What is the current head of state/government for country X?"),

    ("user", "When did Einstein publish the theory of relativity?"),
    ("assistant", "In what year was the (special or general) theory of relativity first published?"),

    ("user", "What's the capital of a specific US state like California?"),
    ("assistant", "What is the capital city of the given US state?"),
]

# --- Build the prompt template (system + few-shot + new question) ---
prompt = PromptTemplate.from_messages(
    [
        ("system",
         "You are an expert at world knowledge. Your task is to step back and "
         "paraphrase a question into a more generic step-back question that is easier to answer. "
         "Respond with only the rephrased generic question."),
        few_shot_prompt,                   # few-shot examples
        ("user", "{question}"),            # new question slot
    ]
)

if __name__ == "__main__":
    question = "What is task decomposition for LLM agents?"
    messages = prompt.format(question=question)

    llm = ChatOllamaMini(model="gemma3:1b", temperature=0.0, base_url="http://localhost:11434")
    step_back = llm.invoke(messages)

    print("STEP-BACK:", step_back)
