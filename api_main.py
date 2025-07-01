import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from ai import (
    CENTRAL_LLM_TOOLS,
    create_central_llm_prompt,
    create_central_llm_with_tools,
)

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://192.168.1.76:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins or ["*"] for all
    allow_credentials=True, # Allows cookies to be included in requests
    allow_methods=["*"],    # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],    # Allows all headers
)
langfuse_client = get_client()

GLOBAL_CHAT_HISTORY: list[BaseMessage] = [
    AIMessage(content="How can I help you today?")
]


class UserQuery(BaseModel):
    text: str


# --- Pydantic Models for Streaming Output (for client rendering) ---
class StreamContent(BaseModel):
    type: str = "content"
    data: str


class StreamToolCallRequest(BaseModel):
    type: str = "ai_tool_request"
    message_id: str
    tool_calls: list[dict[str, Any]]


class StreamToolResponseMessage(BaseModel):
    type: str = "tool_response"
    tool_call_id: str
    tool_name: str
    content: str  # Contains tool result or error message


class StreamFinalAIMessage(BaseModel):
    type: str = "final_ai_response"
    message_id: str
    content: str
    tool_calls: list[dict[str, Any]] = None


class StreamEnd(BaseModel):
    type: str = "stream_end"


@app.post("/chat/reset")
async def chat_reset():
    GLOBAL_CHAT_HISTORY.clear()
    GLOBAL_CHAT_HISTORY.append(AIMessage(content="How can I help you today?"))
    logging.info("Chat history reset.")
    return {"message": "Chat history reset successfully"}


@app.post("/chat/invoke")
async def chat_invoke(query: UserQuery, fastapi_request: Request):
    user_input_text = query.text
    try:
        with langfuse_client.start_as_current_span(
            name="fastapi-chat-request"
        ) as langfuse_parent_span:
            langfuse_parent_span.update_trace(input={"input": user_input_text})
            langfuse_langchain_handler = CallbackHandler()

            user_message = HumanMessage(content=user_input_text)
            GLOBAL_CHAT_HISTORY.append(user_message)

            llm_with_tools = await create_central_llm_with_tools()
            prompt_template = create_central_llm_prompt()
            chain = prompt_template | llm_with_tools

            async def stream_generator() -> AsyncGenerator[str, None]:
                nonlocal langfuse_langchain_handler  # Make handler accessible
                accumulated_initial_ai_message: AIMessage = None

                # Stream 1: Initial LLM response
                async for chunk in chain.astream(
                    {"messages": list(GLOBAL_CHAT_HISTORY)},
                    config={"callbacks": [langfuse_langchain_handler]},
                ):
                    if accumulated_initial_ai_message is None:
                        accumulated_initial_ai_message = chunk
                    else:
                        accumulated_initial_ai_message += chunk
                    if chunk.content:
                        yield f"data: {StreamContent(data=chunk.content).model_dump_json()}\n\n"

                if not accumulated_initial_ai_message:
                    logging.error("No initial AI message received from the chain.")
                    yield f"data: {StreamEnd().model_dump_json()}\n\n"  # Gracefully end stream
                    return

                GLOBAL_CHAT_HISTORY.append(accumulated_initial_ai_message)

                # Process tool calls if any
                if accumulated_initial_ai_message.tool_calls:
                    yield f"data: {StreamToolCallRequest(message_id=accumulated_initial_ai_message.id, tool_calls=accumulated_initial_ai_message.tool_calls).model_dump_json()}\n\n"

                    tool_messages_for_llm: list[ToolMessage] = []
                    for tool_call in accumulated_initial_ai_message.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_function = next(
                            (t for t in CENTRAL_LLM_TOOLS if t.name == tool_name), None
                        )

                        tool_result_content = ""
                        if tool_function:
                            try:
                                tool_result = await tool_function.ainvoke(tool_args)
                                tool_result_content = str(tool_result)
                            except Exception as e:
                                logging.error(f"Error executing tool {tool_name}: {e}")
                                tool_result_content = (
                                    f"Error executing tool {tool_name}: {str(e)}"
                                )
                        else:
                            logging.error(f"Tool '{tool_name}' not found.")
                            tool_result_content = (
                                f"Error: Tool '{tool_name}' not found."
                            )

                        tool_message_obj = ToolMessage(
                            content=tool_result_content, tool_call_id=tool_call["id"]
                        )
                        tool_messages_for_llm.append(tool_message_obj)
                        yield f"data: {StreamToolResponseMessage(tool_call_id=tool_call['id'], tool_name=tool_name, content=tool_result_content).model_dump_json()}\n\n"

                    if tool_messages_for_llm:
                        GLOBAL_CHAT_HISTORY.extend(tool_messages_for_llm)
                        accumulated_final_ai_message: AIMessage = None
                        # Stream 2: LLM response after tool execution
                        async for chunk in chain.astream(
                            {"messages": list(GLOBAL_CHAT_HISTORY)},
                            config={"callbacks": [langfuse_langchain_handler]},
                        ):
                            if accumulated_final_ai_message is None:
                                accumulated_final_ai_message = chunk
                            else:
                                accumulated_final_ai_message += chunk
                            if chunk.content:
                                yield f"data: {StreamContent(data=chunk.content).model_dump_json()}\n\n"

                        if accumulated_final_ai_message:
                            GLOBAL_CHAT_HISTORY.append(accumulated_final_ai_message)
                            yield f"data: {StreamFinalAIMessage(message_id=accumulated_final_ai_message.id, content=accumulated_final_ai_message.content, tool_calls=accumulated_final_ai_message.tool_calls).model_dump_json()}\n\n"
                        else:
                            logging.warning(
                                "No final AI message received after tool calls."
                            )
                else:  # No tool calls, initial AI message is the final one
                    yield f"data: {StreamFinalAIMessage(message_id=accumulated_initial_ai_message.id, content=accumulated_initial_ai_message.content, tool_calls=accumulated_initial_ai_message.tool_calls).model_dump_json()}\n\n"

                yield f"data: {StreamEnd().model_dump_json()}\n\n"

            response = StreamingResponse(
                stream_generator(), media_type="text/event-stream"
            )
            return response

    except ValueError as ve:  # From ai.py, e.g., missing API key
        logging.error(f"ValueError in /chat/invoke: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:  # From ai.py, e.g., MCP tool failure
        logging.error(f"RuntimeError in /chat/invoke: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logging.exception(
            f"Unhandled exception in /chat/invoke: {e}"
        )  # Logs full traceback
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {type(e).__name__}",
        )


# Add uvicorn runner if this file is run directly (for development)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Note: Uvicorn has its own logging. This basicConfig is for app-level logs.

    if not os.environ.get("GOOGLE_API_KEY"):
        logging.error(
            "CRITICAL: GOOGLE_API_KEY not found in environment variables. The application will likely fail to initialize LLM."
        )
        logging.error("Please ensure your .env file is correctly set up and loaded.")

    import uvicorn

    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, reload=True)
