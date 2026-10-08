

import os
import chromadb
from dotenv import load_dotenv
from groq import Groq

# Load environment variables (our Groq API key)
load_dotenv()

def get_chroma_collection():
    """Connect to the ChromaDB collection, rebuilding it from policy_docs/ if it is empty."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="lending_policies")

    if collection.count() == 0:
        print("Policy index is empty - rebuilding from policy_docs/ ...")
        documents, ids = [], []
        for filename in sorted(os.listdir("policy_docs")):
            if filename.endswith(".txt"):
                path = os.path.join("policy_docs", filename)
                with open(path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
                ids.append(filename.replace(".txt", ""))
        collection.add(documents=documents, ids=ids)
        print(f"Indexed {len(documents)} policy documents.")

    return collection

def get_groq_client():
    """Create and return a Groq API client."""
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)

def generate_explanation(applicant_index, X_data, shap_values_array, model, 
                          collection, llm_client, decode_feature_value_fn, 
                          state_mapping=None, top_n=3):
    """
    Full RAG pipeline: given an applicant, generate a plain-English, 
    policy-grounded explanation of their risk prediction, including 
    the actual decision and citations.
    """
    from explainability import get_top_reasons
    
    # Get the actual prediction and probability
    applicant_data = X_data.iloc[[applicant_index]]
    prediction = model.predict(applicant_data)[0]
    probability = model.predict_proba(applicant_data)[0][1]
    decision = "DENIED" if prediction == 1 else "APPROVED"
    
    # Get top SHAP reasons
    top_reasons = get_top_reasons(applicant_index, X_data, shap_values_array, top_n=top_n)
    top_reasons = top_reasons.copy()
    top_reasons['decoded_value'] = top_reasons.apply(
        lambda row: decode_feature_value_fn(row['feature'], row['value'], state_mapping), axis=1
    )
    top_reasons['direction'] = top_reasons['shap_contribution'].apply(
        lambda x: 'INCREASES risk' if x > 0 else 'DECREASES risk'
    )
    
    # Retrieve relevant policy context
    feature_names = top_reasons['feature'].tolist()
    query = f"risk factors related to: {', '.join(feature_names)}"
    retrieval_results = collection.query(query_texts=[query], n_results=2)
    
    retrieved_docs = retrieval_results['ids'][0]
    retrieved_texts = retrieval_results['documents'][0]
    retrieved_context = "\n\n".join([
        f"[SOURCE: {doc_id}]\n{text}" 
        for doc_id, text in zip(retrieved_docs, retrieved_texts)
    ])
    
    reasons_text = "\n".join([
        f"- {row['feature']}: value = {row['decoded_value']}, impact = {row['shap_contribution']:.3f} ({row['direction']})"
        for _, row in top_reasons.iterrows()
    ])
    
    prompt = f"""You are a compliance assistant helping explain a credit decision to a loan applicant.

DECISION: The application was {decision}.
Predicted default risk probability: {probability:.1%}

The applicant's top risk factors, based on our model's analysis, are:
{reasons_text}

IMPORTANT: Each factor above is explicitly labeled as either "INCREASES risk" or 
"DECREASES risk". You MUST use this exact labeled direction for each factor.

Relevant underwriting policy context (each block is labeled with its source document):
{retrieved_context}

Write a clear, plain-English explanation (4-5 sentences) for the applicant. Start by 
clearly stating the decision. Then explain the main factors, grounded strictly in the 
exact values and directions provided above. When you reference a policy concept, add 
a citation in the format [Source: document_name] immediately after that sentence. 
Do not infer or state any specific category, tier, or range not explicitly given. 
Write in a professional, respectful tone."""

    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "decision": decision,
        "probability": float(probability),
        "top_reasons": top_reasons,
        "retrieved_context": retrieved_context,
        "sources_used": list(retrieved_docs),
        "explanation": response.choices[0].message.content
    }