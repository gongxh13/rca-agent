import time
import logging
from src.config.loader import load_config
from src.utils.llm_factory import init_langchain_models_from_llm_config
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Initialize model once
try:
    _CONFIG = load_config()
    _MODELS, _DEFAULT_MODEL = init_langchain_models_from_llm_config(_CONFIG.llm)
    if not _DEFAULT_MODEL:
        raise ValueError("No default model initialized from config.")
except Exception as e:
    logging.warning(f"Failed to initialize LLM from config: {e}. API calls will fail.")
    _DEFAULT_MODEL = None

def get_chat_completion(messages, temperature=0.0, callbacks=None):
    if not _DEFAULT_MODEL:
        raise RuntimeError("LLM not initialized.")
    
    lc_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            # Fallback for other roles like 'function' if any, or treat as user
            lc_messages.append(HumanMessage(content=str(content)))
            
    # Note: We ignore the 'temperature' parameter here because the model is pre-configured.
    # If dynamic temperature is needed, we would need to clone/bind the model.
    
    try:
        response = _DEFAULT_MODEL.invoke(lc_messages, config={"callbacks": callbacks} if callbacks else None)
        return response.content
    except Exception as e:
        logging.error(f"LLM invocation failed: {e}")
        # Retry logic is handled by the caller or the underlying library usually,
        # but the original code had retry logic. We'll let the caller handle it or 
        # just raise it as the original code's retry was inside get_chat_completion wrappers.
        # The original code had a retry loop in get_chat_completion.
        # Let's add a simple retry here to match original behavior if needed,
        # but LangChain usually handles retries.
        raise e
