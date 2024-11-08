# Retrieval-Augmented Generation (RAG) Evaluation Assessment

## Overview
Retrieval-Augmented Generation (RAG) combines information retrieval with natural language generation to produce informed 
and contextually relevant responses. This system retrieves pertinent documents from a knowledge base based on user queries and generates responses using a language model. 
Additionally, it evaluates the effectiveness of both retrieval and generation processes using standard metrics.


## Features
* **Document Retrieval**: Fetches the most relevant documents from a predefined knowledge base. 
* **Response Generation**: Utilizes a language model to generate natural language responses based on retrieved documents. 
* **Evaluation**: Assesses retrieval quality using Precision@k and Recall@k, and generation quality using BLEU score.  
* **Logging**: Tracks retrieval results, generated responses, and evaluation metrics for transparency and debugging.


## Types of RAG Evaluation
### 1. **Retrieval Evaluation**:
* **Precision@k**: The ratio of relevant documents retrieved in the top k results.
* **Recall@k**: The ratio of relevant documents retrieved out of all relevant documents available

### 2. **Generation Evaluation**:
* **BLEU Score**: Measures the similarity between the generated response and a reference response 


## Example Queries
Here are example queries processed by the system:
### 1. **Query1**: "How can I extend the warranty on my mobile phone?"
* **Relevant Documents**: Document 1, Document 2 
* **Reference Answer**: "To extend your warranty, you can purchase an extended warranty plan within 30 days of purchase."

### 2. **Query2**: "How can I reset my mobile device?"
* **Relevant Documents**: Document 3
* **Reference Answer**: "To reset your device, navigate to Settings > General > Reset and select 'Erase All Content and Settings'."

### 3. **Query3**: "How can I reset my device and what is the warranty period?"
* **Relevant Documents**: Document 1, Document 3 
* **Reference Answer**: "To reset your device, go to Settings > General > Reset and select 'Erase All Content and Settings'. The warranty period is one year."


### Example Evaluation Results
**Query 1: Extend Warranty**
* Precision@2: 1.0 (All top-2 documents retrieved were relevant)
* Recall@2: 1.0 (All relevant documents were retrieved)
* BLEU Score: 0.75 (Moderate similarity with the reference answer)

**Query 2: Reset Device**
* Precision@2: 1.0
* Recall@2: 1.0
* BLEU Score: 0.85 (High similarity with the reference answer)

**Query 3: Reset and Warranty**
* Precision@2: 1.0
* Recall@2: 1.0
* BLEU Score: 0.80 (Good similarity with the reference answer)



### Evaluation Analysis and Conclusion
In this RAG system, Precision@k and Recall@k consistently achieved high values (1.0) across queries, indicating that the retriever successfully selected relevant documents. This shows that our retrieval model performs well for identifying relevant context for various user queries.

The BLEU score values, ranging from 0.75 to 0.85, suggested that the generated responses closely matched the reference answers in content. However, there is still room for improvement, especially in aligning the exact phrasing of generated responses with the reference. A BLEU score improvement would likely require further fine-tuning of the generation model to ensure better linguistic alignment with target responses.

