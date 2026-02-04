from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

"""
RAG (Retrieval-Augmented Generation) is a technique that combines retrieval-based and generative models for tasks like question answering. 
It retrieves relevant documents from a knowledge base and uses them to generate more accurate and context-aware responses.

Basic Usage of RAG in Python (using Hugging Face Transformers):

1. Install required libraries:
    pip install transformers datasets

2. Example code:
"""


# Load pre-trained RAG model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
     "facebook/rag-sequence-nq",
     index_name="exact",
     use_dummy_dataset=True  # For demonstration; replace with your own dataset for real use
)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Input question
question = "What is retrieval-augmented generation?"

# Tokenize input
input_dict = tokenizer.prepare_seq2seq_batch([question], return_tensors="pt")

# Generate answer
generated = model.generate(
     input_ids=input_dict["input_ids"],
     attention_mask=input_dict["attention_mask"],
     num_return_sequences=1
)

# Decode and print answer
answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print("Question:", question)
print("RAG Answer:", answer)

"""
Leverage:
- RAG enables models to access external knowledge, improving factual accuracy.
- Useful for open-domain question answering, chatbots, and knowledge-intensive tasks.
- Can be customized with your own document corpus for domain-specific applications.
"""