def answer_questions_directly(user_request: str) -> list[str]:
    """
    When no files are provided, answer the questions directly with the LLM
    and return a JSON array of strings.
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
        answers = json.loads(content)

        # Handle {"answers": [...]} or raw [...]
        if isinstance(answers, dict):
            for v in answers.values():
                if isinstance(v, list):
                    return v
            raise ValueError("Unexpected JSON object from AI")
        elif isinstance(answers, list):
            return answers
        else:
            raise ValueError("Unexpected response format")

    except Exception as e:
        print(f"❌ Error in direct answering: {e}")
        return [f"Error answering questions: {e}"]
