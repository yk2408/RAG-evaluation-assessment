import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from nltk.translate.bleu_score import sentence_bleu

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RAGEvaluation')


class RAGSystem:
    def __init__(self, documents, top_k=2):
        self.documents = documents
        self.top_k = top_k
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator = pipeline("text-generation", model="gpt2")

        # Encode document embeddings
        self.doc_embeddings = self.retriever.encode(documents, convert_to_tensor=True)
        logger.info("Document embeddings created.")

    def retrieve_documents(self, query):
        query_embedding = self.retriever.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.doc_embeddings)
        top_results = torch.topk(cosine_scores, k=self.top_k)

        retrieved_docs = [self.documents[idx] for idx in top_results.indices[0]]
        logger.info(f"User query '{query}'")
        logger.info(f"Retrieved Documents: {retrieved_docs}")
        return retrieved_docs

    def generate_response(self, query, context):
        input_text = f"Question: {query} Context: {context}"
        response = self.generator(input_text, max_new_tokens=500)[0]["generated_text"]
        logger.info(f"Generated Response: {response}")
        return response

    def evaluate_retrieval(self, retrieved_docs, relevant_docs):
        # Precision@k and Recall@k
        def precision_at_k(retrieved_docs, relevant_docs, k):
            retrieved_top_k = retrieved_docs[:k]
            relevant_retrieved = [doc for doc in retrieved_top_k if doc in relevant_docs]
            return len(relevant_retrieved) / k

        def recall_at_k(retrieved_docs, relevant_docs, k):
            retrieved_top_k = retrieved_docs[:k]
            relevant_retrieved = [doc for doc in retrieved_top_k if doc in relevant_docs]
            return len(relevant_retrieved) / len(relevant_docs)

        precision = precision_at_k(retrieved_docs, relevant_docs, self.top_k)
        recall = recall_at_k(retrieved_docs, relevant_docs, self.top_k)

        logger.info(f"Precision@{self.top_k}: {precision:.2f}")
        logger.info(f"Recall@{self.top_k}: {recall:.2f}")
        return precision, recall

    def evaluate_generation(self, reference, generated_response):
        # BLEU Score
        def bleu_score(reference, generated_response):
            reference_tokens = reference.split()
            generated_tokens = generated_response.split()
            score = sentence_bleu([reference_tokens], generated_tokens)
            return score

        bleu = bleu_score(reference, generated_response)
        logger.info(f"BLEU Score: {bleu:.2f}")
        return bleu

    def process_query(self, query, relevant_docs, reference):
        # Retrieve
        retrieved_docs = self.retrieve_documents(query)

        # Generate
        context = " ".join(retrieved_docs)
        generated_response = self.generate_response(query, context)

        # Evaluate
        retrieval_metrics = self.evaluate_retrieval(retrieved_docs, relevant_docs)
        generation_metric = self.evaluate_generation(reference, generated_response)

        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_response": generated_response,
            "retrieval_metrics": retrieval_metrics,
            "generation_metric": generation_metric
        }


# Example Documents
documents = [
    "Document 1: Our mobile devices come with a one-year warranty that covers manufacturer defects.",
    "Document 2: You can extend your warranty by purchasing an extended warranty plan within 30 days of purchase.",
    "Document 3: To reset your mobile device, go to Settings > General > Reset and select 'Erase All Content and Settings'.",
    "Document 4: Battery life varies depending on usage, but most devices last around 8-10 hours of active use.",
    "Document 5: Our devices support wireless charging and come with a fast-charging USB-C cable in the box."
]

# Queries and Evaluation Data
queries = [
    {
        "query": "How can I extend the warranty on my mobile phone?",
        "relevant_docs": [
            "Document 1: Our mobile devices come with a one-year warranty that covers manufacturer defects.",
            "Document 2: You can extend your warranty by purchasing an extended warranty plan within 30 days of purchase."
        ],
        "reference": "To extend your warranty, you can purchase an extended warranty plan within 30 days of purchase."
    },
    {
        "query": "How can I reset my mobile device?",
        "relevant_docs": [
            "Document 3: To reset your mobile device, go to Settings > General > Reset and select 'Erase All Content and Settings'."
        ],
        "reference": "To reset your mobile device, navigate to Settings > General > Reset, then select 'Erase All Content and Settings'."
    },
    {
        "query": "How can I reset my device and what is the warranty period?",
        "relevant_docs": [
            "Document 1: Our mobile devices come with a one-year warranty that covers manufacturer defects.",
            "Document 3: To reset your mobile device, go to Settings > General > Reset and select 'Erase All Content and Settings'."
        ],
        "reference": "To reset your device, go to Settings > General > Reset and select 'Erase All Content and Settings'. The device comes with a one-year warranty covering manufacturer defects."
    }
]

if __name__ == '__main__':

    # Initialize RAG System
    rag_system = RAGSystem(documents)

    # Process each query
    results = []
    for q in queries:
        result = rag_system.process_query(q["query"], q["relevant_docs"], q["reference"])
        results.append(result)
        logger.info(f"Results: {result}")
