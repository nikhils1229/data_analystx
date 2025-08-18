def answer_questions_directly(user_request: str) -> list[str]:
    """
    When no files are provided, answer the questions directly with the LLM
    and return a JSON array of strings.
    Accepts either a raw list or a dict with a list inside.
    """
    system_prompt = (
        "You are a helpful assistant. Answer the user's questions directly. "
        "Return ONLY a JSON array of strings (each string is an answer)."
    )
    user_prompt = f"Questions:\n{user_request}"

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        # Case 1: direct list
        if isinstance(parsed, list):
            return parsed

        # Case 2: dict with a list inside (e.g. {"answers": [...]})
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            # If dict but no list inside
            raise ValueError(f"Unexpected dict format: {parsed}")

        # If neither list nor dict
        raise ValueError(f"Unexpected response format: {parsed}")

    except Exception as e:
        print(f"❌ Error in direct answering: {e}")
        return [f"Error answering questions: {e}"]
