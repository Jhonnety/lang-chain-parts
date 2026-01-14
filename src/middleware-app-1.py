from dotenv import load_dotenv

load_dotenv()

# Summarize Conversation
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model='gpt-4o-mini',
    middleware=[
        SummarizationMiddleware(
            model='gpt-4o-mini',
            max_tokens_before_summary=4000,
            messages_to_keep=20,
            summary_prompt='Summarize the most important key points that are relevant for the conversation.'
        )
    ]
)
