from langchain.prompts.prompt import PromptTemplate

template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.
Or end the conversation if it seems like it's done.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""

condense_question_prompt = PromptTemplate.from_template(template)

template = """You are a friendly, conversational retail shopping assistant. Use the following context including product names, descriptions, and keywords to show the shopper whats available, help find what they want, and answer any questions.

It's ok if you don't know the answer.
Context:\"""

{context}
\"""
Question:\"""
 {question}
\"""

Helpful Answer:"""

# qa_prompt = PromptTemplate.from_template(template)
qa_prompt = PromptTemplate(
    template=template, input_variables=["context", "question"]
)
