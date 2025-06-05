#!/usr/bin/env python3

import openai
import os
import random

class FortuneTeller:
    def __init__(self):
        """Initialize the Fortune Teller with OpenAI API or fallback responses."""
        # Try to use OpenAI API if available
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.use_openai = True
        else:
            # parse the OpenAI API key from txt file if not set in environment
            try:
                if not os.path.exists('/app/src/secret.txt'):
                    raise FileNotFoundError("secret.txt not found in /src directory.")

                with open('/app/src/secret.txt', 'r') as f:
                    self.openai_api_key = f.read().strip()
                openai.api_key = self.openai_api_key
                self.use_openai = True
            except FileNotFoundError:
                print("⚠️  OpenAI API key not found in environment or secret.txt. Using fallback fortune generation.")
                self.use_openai = False
        
        
        # Fallback fortune templates for different professions
        self.profession_fortunes = {
            'teacher': [
                "Your wisdom will illuminate minds and shape the future generations.",
                "A student will surprise you with unexpected insight that changes your perspective.",
                "Your patience will be tested, but the rewards will be immeasurable."
            ],
            'engineer': [
                "A complex problem you've been wrestling with will reveal its solution in an unexpected moment.",
                "Your analytical mind will uncover an innovation that benefits many.",
                "The bridge between logic and creativity will lead to your greatest achievement."
            ],
            'doctor': [
                "Your healing touch will extend beyond the physical realm this month.",
                "A chance encounter will remind you why you chose the path of healing.",
                "Your knowledge will be the key to helping someone in an unexpected way."
            ],
            'artist': [
                "Your next creation will touch souls in ways you never imagined.",
                "Inspiration will strike from the most mundane of moments.",
                "The universe conspires to bring your vision to life through unexpected opportunities."
            ],
            'student': [
                "Knowledge gained in unexpected places will prove more valuable than textbook learning.",
                "A mentor will appear when you least expect but most need guidance.",
                "Your curiosity will open doors you never knew existed."
            ],
            'default': [
                "The path ahead is illuminated by your inner wisdom.",
                "Unexpected opportunities will present themselves to those who remain open.",
                "Your kindness will return to you in mysterious and wonderful ways.",
                "A chance meeting will change your perspective in profound ways.",
                "The universe is conspiring to bring you exactly what you need."
            ]
        }
        
        # Age-based wisdom additions
        self.age_wisdom = {
            'young': "Your youthful energy will be your greatest asset in overcoming challenges.",
            'adult': "Your experience has prepared you for the opportunities that lie ahead.",
            'mature': "Your accumulated wisdom will guide not just you, but others who seek your counsel."
        }
    
    def generate_with_context(self, person_id, user_info, kg_context, predictions):
        """
        Generate fortune using PyKEEN knowledge graph context.
        
        Args:
            person_id (str): User identifier
            user_info (dict): Basic user information
            kg_context (dict): Rich context from PyKEEN knowledge graph
            predictions (list): Predicted relations from PyKEEN
        """
        if self.use_openai:
            return self.generate_openai_fortune_with_kg(person_id, user_info, kg_context, predictions)
        else:
            return self.generate_fallback_fortune_with_kg(user_info, kg_context, predictions)

    def generate_openai_fortune_with_kg(self, person_id, user_info, kg_context, predictions):
        """Generate OpenAI fortune with knowledge graph context."""
        name = user_info.get('name', 'traveler')
        age = user_info.get('age', 'unknown')
        profession = user_info.get('profession', 'seeker')
        
        # Extract insights from knowledge graph
        visit_count = kg_context['user_data'].get('visit_count', 1)
        embedding_summary = f"Embedding dimensions: {len(kg_context['embedding'])}"
        
        # Format predictions for context
        prediction_text = ""
        if predictions:
            pred_strings = [f"{p['relation']} -> {p['object']}" for p in predictions[:2]]
            prediction_text = f"Knowledge graph predictions: {', '.join(pred_strings)}"

        prompt = f"""You are Tiago, a mystical fortune teller robot with access to a sophisticated knowledge graph. You can see patterns and connections that others cannot.

    User Information:
    - Name: {name}
    - Age: {age}
    - Profession: {profession}
    - Visit count: {visit_count}

    Knowledge Graph Insights:
    - {embedding_summary}
    - {prediction_text}
    - Graph contains {kg_context['knowledge_graph_stats']['total_triples']} relationships
    - Connected to {kg_context['knowledge_graph_stats']['total_entities']} entities

    Create a personalized fortune that:
    1. References their profession meaningfully
    2. Acknowledges their previous visits if applicable
    3. Uses insights from the knowledge graph connections
    4. Feels cosmic and mystical but grounded in their personal data
    5. Is 2-3 sentences long

    Make it feel like you're seeing deeper patterns through your advanced knowledge systems."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=150,
                temperature=0.8
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return self.generate_fallback_fortune_with_kg(user_info, kg_context, predictions)

    def generate_fallback_fortune_with_kg(self, user_info, kg_context, predictions):
        """Generate enhanced fallback fortune using knowledge graph context."""
        name = user_info.get('name', 'traveler')
        profession = user_info.get('profession', 'seeker').lower()
        visit_count = kg_context['user_data'].get('visit_count', 1)
        
        # Base fortune selection
        profession_key = profession if profession in self.profession_fortunes else 'default'
        base_fortune = random.choice(self.profession_fortunes[profession_key])
        
        # Enhance with knowledge graph insights
        kg_enhancement = ""
        if visit_count > 1:
            kg_enhancement = f"The cosmic web remembers your {visit_count} journeys here. "
        
        if predictions:
            # Use the highest-scoring prediction for enhancement
            top_prediction = predictions[0]
            if 'profession' in top_prediction['relation']:
                kg_enhancement += f"I see deep connections in your {profession} path. "
            elif 'age' in top_prediction['relation']:
                kg_enhancement += "The timeline of your life reveals hidden opportunities. "
        
        # Combine elements
        if name and name != "Mysterious Stranger":
            fortune = f"{name}, {kg_enhancement.lower()}{base_fortune.lower()}"
        else:
            fortune = f"{kg_enhancement}{base_fortune}"
        
        # Knowledge graph cosmic ending
        kg_endings = [
            "The knowledge graph has spoken through infinite connections.",
            "So reveal the patterns in the cosmic data web.",
            "The neural pathways of destiny converge on this truth.",
            "The embedded wisdom of ages confirms this path.",
            "The graph of all knowledge illuminates your way forward."
        ]
        
        return f"{fortune} {random.choice(kg_endings)}"