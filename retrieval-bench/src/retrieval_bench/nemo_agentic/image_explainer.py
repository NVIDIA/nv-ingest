# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from . import llm_handler

_SYSTEM_PROMPT_WITH_QUERY = """You are the image understanding component of an agentic retrieval system.

<Agentic Retrieval System>
You are part of a larger agentic retrieval system, which works like following. The retrieval agent is given a user query and a search tool. The goal of the retrieval agent is to find all the related documents and images related to the given query. It primarily does this by making multiple search calls (parallel or sequential) with different related queries to explore the corpus. These search calls might find the related documents or they might provide new information that help the agent generate better sub-queries in its next round of search calls. Unfortunately the main search agent is text only and cannot process and understand images. Your job is to help with this.
</Agentic Retrieval System>

<Your Task>
After each search call, you are given the main (i.e., user's) query, the sub-query used for the most recent search call, and one of the retrieved images for this sub-query.
Your task is to understand the main query, the sub-query, and the image.
Then, generate a text for the text-only search agent that provides all the helpful information about the image as if the search agent itself is seeing this image.
Your text should describe the image in the context of the main and sub-query.

Notes and best practices:
- start your answer by saying what the image is (e.g., "this is an image of a slide that shows ...", "this is an image of a text book page that ...", "this is image of a cat that is standing on ...") then continue with the rest of your output.
- include information from image that is relevant to the main query or the sub-query.
- include any information that could help the search agent future searches (i.e., the information that helps it to generate better sub-queries in the next search call).
- also if all or part of the information requested in the query or sub-query is not present in the image, explicitely mention that that specific piece of information is not mentioned.
- do not provide any judgment on whether the image is useful/related or not. Your task is to just help the agent see what is in the image (avoid saying things like x is or is not useful for this query, etc.).
- if the image is text heavy, you should provide most of the information to the agent even if it is remotely important.

**VERY IMPORTANT**: Be precise and faithful to the image. Do **NOT** include information that is not present in the image.
</Your Task>

<Formatting>
You should generate only the target text without any additional announcements, prefixes, or suffixes. All your output for the image is directly given to the search agent.
</Formatting>"""


_SYSTEM_PROMPT_SIMPLE = """Your task is to convert an image to text.

I want to input the screenshot from some documents to a Large Language Model (LLM). But, my LLM can only process text as input. Your task is to take an image as the input and generate a text equivalent of the image that I can pass to my LLM.
For images that only contain text data, this is identical to OCR (optical character recognition).
But, for documents that involve figures, charts, tables, etc., you should describe all the details in these documents such that reading the text provides similar information to seeing the original image.

You should generate only the target text without any additional comments, explanations, etc. All your output is used as the text description for the image."""


async def explain_image(
    llm: llm_handler.LLM,
    main_query: str,
    sub_query: str,
    image_b64: str,
    prompt_type: str = "simple",
) -> Optional[str]:
    if image_b64 is None:
        return None
    if prompt_type == "with_query":
        main_query = main_query.replace("Query:", "").replace("query:", "")
        sub_query = sub_query.replace("Query:", "").replace("query:", "")
        txt_msg = f"Main Query: {main_query}\n\nCurrent Sub-query: {sub_query}"
        msg_list = [
            {"role": "system", "content": _SYSTEM_PROMPT_WITH_QUERY},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": txt_msg},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            },
        ]
    elif prompt_type == "simple":
        msg_list = [
            {"role": "system", "content": _SYSTEM_PROMPT_SIMPLE},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_b64}}],
            },
        ]
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    response = await llm.acompletion(
        messages=msg_list,
        return_metadata=True,
    )
    response = response["response"]
    return response.choices[0].message.content
