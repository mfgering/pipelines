from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import openai
from openai import OpenAI


class Pipeline:
    class Valves(BaseModel):
        openai_api_key: str = None

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "mfg_pipeline_example"

        # The name of the pipeline.
        self.name = "MFG Pipeline Example"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.client = OpenAI()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        print(user)

        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        print(f"outlet:{__name__}")

        print(body)
        print(user)

        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        # If you'd like to check for title generation, you can add the following check
        if body.get("title", False):
            print("Title Generation Request")
        # The body['messages'] includes the system prompt from the model. If the request
        # has a file, then the RAG template is included in the system prompt.

        print(messages)
        print(user_message)
        print(body)
        # Set the API key using values from the valves.
        openai.api_key = self.valves.openai_api_key
        client = self.client
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "you are a helpful assistant"
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "write me a haiku"
                    }
                ]
                },
                # {
                # "role": "assistant",
                # "content": [
                #     {
                #     "type": "text",
                #     "text": "Whispers of the breeze,  \nLeaves dance in the golden light,  \nAutumn's gentle sigh."
                #     }
                # ]
                # }
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )
        #completion = self.client.chat.completions.create(model=model_id, messages=messages)
        content = response.choices[0].message.content
        return content
#        return f"{__name__} response to: {user_message}"
