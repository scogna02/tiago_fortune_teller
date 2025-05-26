import openai
import os

class FortuneTeller:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        #openai.api_key = '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'

    def generate(self, person_id, context):
        prompt = f"You are a mystical fortune teller. Person: {person_id}. Context: {context}."
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
