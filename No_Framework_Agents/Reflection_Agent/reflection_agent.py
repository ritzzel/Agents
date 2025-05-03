from google import genai
from dotenv import load_dotenv
from utils import completion, chat_completion, prompt_struct, chat_append
import os
import logging

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# System prompts for generation and reflection
BASE_GENERATION_SYSTEM_PROMPT = """
Your task is to Generate the best content possible for the user's request.
If the user provides critique, respond with a revised version of your previous attempt.
You must always output the revised content.
"""

BASE_REFLECTION_SYSTEM_PROMPT = """
You are tasked with generating critique and recommendations to the user's generated content.
If the user content has something wrong or something to be improved, output a list of recommendations
and critiques. If the user content is ok and there's nothing to change, output this: <OK>
"""

class ReflectionAgent:
    """
    ReflectionAgent loops through generating content and refining it based on AI-generated feedback.
    This helps reduce hallucinations and improves the quality of responses.
    """

    def __init__(self, model:str=os.getenv('GEMINI_MODEL_NAME')):
        self.model = model
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or not model:
            raise EnvironmentError("Missing required environment variables: GEMINI_API_KEY or GEMINI_MODEL_NAME")
        try:
            self.client = genai.Client(
                api_key=api_key,
                # project=os.getenv('PROJECT_NO')
            )
            self.chat = self.client.chats.create(model=self.model)
        except Exception as e:
            logging.error("Failed to initialize Gemini client: %s", str(e))
            raise

    def _invoke_completion(self, hist:list) -> str:
        """Calls chat completion using the chat object and conversation history."""
        try:
            response = chat_completion(self.chat, self.model, hist)
            return response
        except Exception as e:
            logging.error("Error during chat completion: %s", str(e))

    def generate_response(self, generation_hist:list):
        """Generates a response based on current generation history."""
        response = self._invoke_completion(generation_hist)
        return response

    def reflect(self, reflection_hist:list):
        """Generates a critique for the previously generated content."""
        response = self._invoke_completion(reflection_hist)
        return response

    def invoke_agent(self, user_msg, step):
        """
        Main loop to alternate between generation and reflection steps.

        Args:
            user_msg (str): User's input prompt.
            step (int): Number of reflection cycles.

        Returns:
            str: Final response after reflection.
        """
        generation_hist = [prompt_struct(role='system', message=BASE_GENERATION_SYSTEM_PROMPT),
                           prompt_struct(role='user', message=user_msg)]
        reflection_hist = [prompt_struct(role='system', message=BASE_REFLECTION_SYSTEM_PROMPT)]

        for i in range(step):
            print(f"Iteration {i+1}/{step}")
            logging.info(f"Iteration {i + 1}/{step}")

            #Generate Response
            generation = self.generate_response(generation_hist)
            chat_append(generation_hist, 'assistant', generation.text)
            chat_append(reflection_hist, 'user', generation.text)

            #Reflect on generation
            reflection = self.reflect(reflection_hist)
            if '<OK>' in reflection.text:
                print('No criticism suggested, reflection stopped!')
                logging.info('No criticism suggested, reflection stopped!')
                return generation

            # Append feedback to both histories for next round
            chat_append(generation_hist, 'assistant', reflection.text)
            chat_append(reflection_hist, 'user', reflection.text)

if __name__ == "__main__":
    prompt = 'calculate the grain of sands in universe?'
    agent = ReflectionAgent()
    resp = agent.invoke_agent(prompt, 3)
    print(f'Prompt: {prompt} \nResponse: {resp.text}')




