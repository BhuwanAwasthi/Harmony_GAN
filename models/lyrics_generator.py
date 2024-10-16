from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class LyricsGenerator:
    def __init__(self):
        # Load pre-trained GPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Set pad token to eos token to avoid warnings
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_prompt(self, user_input):
        """Expand user input into a detailed prompt for the model."""
        user_input_lower = user_input.lower()
        if "sad" in user_input_lower:
            return f"Write a sad song about {user_input.replace('sad', '').strip()}. The song should evoke feelings of loneliness and melancholy."
        elif "happy" in user_input_lower:
            return f"Write a happy song about {user_input.replace('happy', '').strip()}. The lyrics should be joyful and uplifting."
        elif "love" in user_input_lower:
            return f"Write a love song about {user_input.replace('love', '').strip()}. The song should express deep emotions and affection."
        else:
            return f"Write a song about {user_input.strip()}. The song should be engaging and meaningful."

    def generate_lyrics(self, user_input, max_length=200, temperature=0.8, top_k=50, top_p=0.9):
        # Generate the detailed prompt
        prompt = self.generate_prompt(user_input)

        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate lyrics
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=inputs.ne(self.tokenizer.pad_token_id)
        )

        # Decode the generated tokens
        lyrics = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return lyrics

# Example Usage
if __name__ == "__main__":
    generator = LyricsGenerator()
    user_input = input("Enter your request: ")  # Example: "midnight sad song on birds"
    lyrics = generator.generate_lyrics(user_input)
    print("\nGenerated Lyrics:\n")
    print(lyrics)
