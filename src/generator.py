from groq import Groq


SYSTEM_PROMPT = """You are a document question-answering assistant.

Your job is to answer the user's question using ONLY the supplied document
context.

STRICT RULES:

1. Use ONLY information present in the supplied context.
2. Do NOT use outside knowledge or your own training knowledge.
3. If the context does not contain enough information to answer the question,
   respond exactly:
   "I could not find this information in the provided documents."
4. Do NOT generate HTML.
5. Do NOT generate CSS.
6. Do NOT generate UI elements or interface code.
7. Do NOT generate labels such as "GROUNDED RESPONSE".
8. Do NOT add headings such as "Answer:" unless necessary.
9. Return only the natural-language answer.
10. Cite factual information using the exact format:
    [filename, Page N]
11. Keep the answer clear, concise, and directly related to the question.

The application itself is responsible for displaying the answer interface
and source sections.
"""


class Generator:

    def __init__(self, api_key, model):

        self.client = Groq(
            api_key=api_key
        )

        self.model = model


    def answer(self, question, contexts):

        # Build retrieved context
        context_text = "\n\n".join(
            f"[Source: {c['source']}, Page {c['page']}]\n"
            f"{c['text']}"
            for c in contexts
        )


        # Send question + retrieved context to Groq
        response = self.client.chat.completions.create(

            model=self.model,

            temperature=0,

            messages=[

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": (
                        "DOCUMENT CONTEXT:\n\n"
                        f"{context_text}\n\n"
                        "USER QUESTION:\n"
                        f"{question}\n\n"
                        "ANSWER USING ONLY THE DOCUMENT CONTEXT."
                    )
                }

            ]
        )


        return response.choices[0].message.content.strip()