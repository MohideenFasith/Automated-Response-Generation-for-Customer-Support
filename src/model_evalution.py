from datasets import load_metric
from model_training import train_model
from data_preprocessing import load_and_preprocess_data

def evaluate_model(predictions, references):
    rouge = load_metric('rouge')
    bleu = load_metric('bleu')
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    return rouge_score, bleu_score

if __name__ == "__main__":
    # Load dataset and model
    dataset = load_and_preprocess_data()
    model, tokenizer = train_model()

    # Generate predictions
    def generate_response(input_text):
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    preds = [generate_response(query) for query in dataset['test']['query']]
    refs = [response for response in dataset['test']['response']]

    # Evaluate the model
    rouge_score, bleu_score = evaluate_model(preds, refs)
    print("ROUGE Score:", rouge_score)
    print("BLEU Score:", bleu_score)
