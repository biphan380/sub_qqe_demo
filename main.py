import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

import nest_asyncio
nest_asyncio.apply()

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import ServiceContext

# Using the LlamaDebugHandler to print the trace of the sub questions 
# captured by the SUB_QUESTION callback event type

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

# load data
pg_essay = SimpleDirectoryReader(input_dir="data/").load_data()

# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    pg_essay, use_async=True, service_context=service_context
).as_query_engine()

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="pg_essay", description="Paul Graham essay on What I Worked On"
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
)

response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)

print(response)

# iterate through sub_question items captured in SUB_QUESTION event

from llama_index.callbacks.schema import CBEventType, EventPayload

for i, (start_event, end_event) in enumerate(
    llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
):
    qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
    print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
    print("Answer: " + qa_pair.answer.strip())
    print("====================================")