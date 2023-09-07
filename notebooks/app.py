import os
import openai
from langchain.chat_models import ChatOpenAI
import chainlit as cl

from langchain.chains import LLMChain, SequentialChain 
from langchain.prompts import PromptTemplate

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)

from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]


from llama_index import SimpleDirectoryReader

required_exts = [".txt"]

reader = SimpleDirectoryReader(
    input_dir="../data",
    required_exts=required_exts,
    recursive=True,
    filename_as_id=True
)

docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

chatgpt = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=chatgpt, chunk_size=1024)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)

doc_summary_index = DocumentSummaryIndex.from_documents(
    docs,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
)

#print(doc_summary_index.get_document_summary('../data/final-hh.txt'))

@cl.on_chat_start
async def factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=True,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=512,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    
    generation_type = None

    twitter_expert_prompt = "You are a twitter expert. Create a twitter thread based on this: "
    
    summary_text = doc_summary_index.get_document_summary('../data/final-hh.txt')
    summary_text_without_newlines = summary_text.strip()
    prompt = twitter_expert_prompt + summary_text
    
    twitter_prompt = twitter_expert_prompt + summary_text_without_newlines
    twitter_prompt_template = PromptTemplate.from_template(
        twitter_expert_prompt
    )
    
    while generation_type == None:
        generation_type = await cl.AskUserMessage(
            content="Twitter or blog?", timeout=15
        ).send()
        
        if generation_type == 'twitter':
            print("twitter generation...")
        elif generation_type == 'blog':
            print("blog generation...")
        
    cl.Message(content=f"generating {generation_type['content']} thread").send()
    
    
    
    prompt = 'write a tweet about' + summary_text
    print(f'prompt: {prompt}')
    
    
    llm_twitter_expert = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.7)
    tweet_thread_chain = LLMChain(llm=llm_twitter_expert,prompt=twitter_prompt_template) 
    #tweet_thread = tweet_thread_chain.run(prompt=prompt)
    #await cl.Message(content=tweet_thread).send()
    
    the_final_prompt = f"write a twitter thread about {summary_text_without_newlines}"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a twitter expert."},
            #{"role": "assistant", "content": "{summary_text_without_newlines}"},
            {"role": "user", "content": the_final_prompt},
        ]
    )
    await cl.Message(content=response['choices'][0]['message']['content']).send()
    
    
    
    #await cl.Message(content=f"generating {doc_summary_index.get_document_summary('../data/final-hh.txt')}").send()
 



