from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

def get_qa_pipeline(model_name="distilbert-base-cased-distilled-squad"):
    """
    Loads a pre-trained QA model and returns a pipeline.
    DistilBERT is used for CPU performance.
    """
    print(f"Initializing QA Pipeline with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipe

def answer_question(context, question, qa_pipe):
    """
    Uses the model to extract an answer from the context.
    """
    result = qa_pipe(question=question, context=context)
    return result['answer'], result['score']
