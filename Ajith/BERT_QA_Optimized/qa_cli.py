import argparse
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Question Answering")

    parser.add_argument(
        "--model",
        default="./qa_model",
        help="Path to trained QA model"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="Context paragraph"
    )

    args = parser.parse_args()

    qa_pipeline = pipeline(
        "question-answering",
        model=args.model,
        tokenizer=args.model
    )

    result = qa_pipeline(
        question=args.question,
        context=args.context
    )

    print("\nAnswer:", result["answer"])
    print("Confidence Score:", round(result["score"], 4))

if __name__ == "__main__":
    main()
