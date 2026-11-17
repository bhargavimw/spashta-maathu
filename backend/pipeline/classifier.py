"""Stage 4: LLM Disfluency Classifier and Policy Engine.

Prompts an LLM in strict JSON mode to select kept word indices and pause overrides.
Validates LLM output against index bounds, ascending order, and uniqueness.
Falls back to deterministic rule-based cleaning on validation or API failure.
Never modifies or synthesizes words; clean text is derived solely from kept words.
"""
