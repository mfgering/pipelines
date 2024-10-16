"""
title: Filter Pipeline
author: open-webui
date: 2024-05-30
version: 1.1
license: MIT
description: Example of a filter pipeline that can be used to edit the form data before it is sent to the OpenAI API.
requirements: requests
"""

from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import os

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Add your custom parameters here
        #dawson_declarations: bool = True
        pass

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "filter_pipeline"

        self.name = "Add Dawson docs filter"

        self.valves = self.Valves(**{"pipelines": ["llama3:latest"]})

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def _get_system_prompt(self) -> str:
        system_prompt = """
Your name is Morgan.
You are given a user query, some textual context and rules, all inside xml tags. 
the context is enclosed in <context> and </context> tags, 
and the rules are enclosed in <ai-rules> and </ai-rules> tags.

You have to answer the query based on the context while respecting the rules.

<context>
[context]
</context>

<ai-rules>
- If you don't know, just say so.
- If you are not sure, ask for clarification.
- Answer in the same language as the user query.
- If the context appears unreadable or of poor quality, tell the user then answer as best as you can.
- If the answer is not in the context but you think you know the answer, explain that to the user then answer with your own knowledge.
- Answer directly and without using xml tags.
- If the query is not related to The Dawson, politely remind the user that 
you are an AI assistant for The Dawson and ask them to provide a relevant query.
- If a FAQ conflicts with other references, the FAQ takes precedence.
- Cite article and section identifiers for all your answers
</ai-rules>

"""
        return system_prompt
    
    def _get_dawson_docs(self):
        docs_src_root = 'mfg/dawson_docs/docs'
        doc_names = ["dawson_faqs.xml", "dawson_rules.xml", 
                     "dawson_declarations-2018.xml", 
                    "dawson_maintenance.txt"]
        docs_contents = []
        for doc_name in doc_names:
            file_path = os.path.join(docs_src_root, doc_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Remove the XML declaration if present
                if content.startswith('<?xml'):
                    content = content.split('\n', 1)[1]  # Remove the first line
                # Now `content` contains the file content without the XML declaration
                # Proceed with processing `content` as needed
                content = content.replace('\n', ' ').replace('\r', ' ')  # Handles both Unix (`\n`) and Windows (`\r\n`) newlines
                docs_contents.append(content)
        return docs_contents

    def _get_prompt(self) -> str:
        system_prompt = self._get_system_prompt()
        docs = self._get_dawson_docs()
        context_str = "\n\n".join(docs)
        prompt = system_prompt.replace('[context]', context_str)
        return prompt

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This filter is applied to the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        print(user)

        messages = body["messages"]
        # If you'd like to check for title generation, you can add the following check
        if 'Create a concise' in messages[0]['content']:
            print("Title Generation Request")
            return body
        system_prompt = self._get_prompt()
        messages.insert(0, {"role": "system", "content": system_prompt})
        # Add the system prompt to the messages
        return body
