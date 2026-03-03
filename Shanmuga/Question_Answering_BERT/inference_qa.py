import sys
from model_qa import get_qa_pipeline, answer_question

def main():
    print("--- BERT Question Answering System ---")
    qa_pipe = get_qa_pipeline()
    
    while True:
        print("\n" + "="*50)
        context = input("Enter the Context (or type 'exit' to quit): ")
        if context.lower() == 'exit':
            break
            
        question = input("Enter your Question: ")
        
        print("\nAnalyzing...")
        answer, confidence = answer_question(context, question, qa_pipe)
        
        print(f"\nModel's Answer: {answer}")
        print(f"Confidence Score: {confidence:.4f}")

if __name__ == "__main__":
    main()
