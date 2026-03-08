from model_qa import get_qa_pipeline, answer_question

def test_inference():
    qa_pipe = get_qa_pipeline()
    
    context = "The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America."
    question = "What is another name for the Amazon rainforest?"
    
    print(f"\nContext: {context}")
    print(f"Question: {question}")
    
    answer, confidence = answer_question(context, question, qa_pipe)
    
    print(f"\nModel's Answer: {answer}")
    print(f"Confidence: {confidence:.4f}")
    
    assert "Amazonia" in answer or "Amazon Jungle" in answer
    print("\nInference Test Passed!")

if __name__ == "__main__":
    test_inference()
