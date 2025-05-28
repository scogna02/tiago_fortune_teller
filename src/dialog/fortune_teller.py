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
            self.use_openai = False
            print("⚠️  OpenAI API key not found. Using fallback fortune generation.")
        
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
    
    def get_age_category(self, age):
        """Categorize age for fortune generation."""
        try:
            age_num = int(age)
            if age_num < 25:
                return 'young'
            elif age_num < 55:
                return 'adult'
            else:
                return 'mature'
        except (ValueError, TypeError):
            return 'adult'  # default
    
    def generate_openai_fortune(self, person_id, user_info, context):
        """Generate fortune using OpenAI API."""
        try:
            name = user_info.get('name', 'traveler')
            age = user_info.get('age', 'unknown')
            profession = user_info.get('profession', 'seeker')
            
            prompt = f"""You are a mystical, wise fortune teller robot named Tiago. You speak with cosmic wisdom and mystical flair, but your fortunes are personalized, positive, and inspiring.

User Information:
- Name: {name}
- Age: {age}
- Profession: {profession}
- Context: {context}

Create a personalized fortune that:
1. References their profession in a meaningful way
2. Considers their age/life stage appropriately
3. Is positive and inspiring
4. Has a mystical, cosmic tone
5. Is 2-3 sentences long
6. Feels personalized to them specifically

Be mystical but not vague - make it feel truly personalized to this individual."""

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=200,
                temperature=0.8
            )
            return response['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return self.generate_fallback_fortune(user_info)
    
    def generate_fallback_fortune(self, user_info):
        """Generate fortune using local templates when OpenAI is unavailable."""
        name = user_info.get('name', 'traveler')
        age = user_info.get('age', 'unknown')
        profession = user_info.get('profession', 'seeker').lower()
        
        # Get profession-specific fortune
        profession_key = profession if profession in self.profession_fortunes else 'default'
        base_fortune = random.choice(self.profession_fortunes[profession_key])
        
        # Add age-specific wisdom
        age_category = self.get_age_category(age)
        age_addition = self.age_wisdom.get(age_category, "")
        
        # Personalize with name
        if name and name != "Mysterious Stranger":
            fortune = f"{name}, {base_fortune.lower()} {age_addition}"
        else:
            fortune = f"{base_fortune} {age_addition}"
        
        # Add cosmic flair
        cosmic_endings = [
            "The stars have spoken.",
            "So say the cosmic forces.",
            "The universe has decreed it.",
            "This is written in the celestial tapestry.",
            "The cosmic winds carry this truth to you."
        ]
        
        return f"{fortune} {random.choice(cosmic_endings)}"
    
    def generate(self, person_id, user_info, context=None):
        """
        Generate a personalized fortune based on user information.
        
        Args:
            person_id (str): Unique identifier for the person
            user_info (dict): Dictionary containing name, age, profession
            context: Knowledge graph embedding or additional context
            
        Returns:
            str: Personalized fortune message
        """
        if self.use_openai:
            return self.generate_openai_fortune(person_id, user_info, context)
        else:
            return self.generate_fallback_fortune(user_info)



"""import openai
import os

class FortuneTeller:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = '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'

    def generate(self, person_id, context):
        prompt = f"You are a mystical fortune teller. Person: {person_id}. Context: {context}."
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        #return response['choices'][0]['message']['content']
        return "Your simulated fortune: Success is in your future!"""




