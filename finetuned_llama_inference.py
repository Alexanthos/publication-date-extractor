from unsloth import FastLanguageModel, get_chat_template
import ast
import argparse
import pandas as pd

def format_question(context:str,
                    prompt:str = 'What is the publication date of the document? Output as a structured JSON object with a format DD/MM/YYYY.'):
    return [{'role': 'user', 'content': f'Beggining and end of the document :\n{context}\n{prompt}'}]


def load_lora_model_inference(checkpoint):
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
    return model, tokenizer

def format_predicted_date(date_str):
    """
    date_str: str, a string containing a date in the format '{'predicted_date' : 'DD/MM/YYYY'}' into 'DD/MM/YYYY'.
                    can also handle '{'predicted_date' : 'YYYY'}'
    """
    try:
        return ast.literal_eval(date_str)['predicted_date']
    except SyntaxError as e:
        try:
            return ast.literal_eval(date_str[:27])['predicted_date']
        except:
            print(e)
            print(date_str)
            return date_str

def predict_date(message, model, tokenizer):
    """
    Predicts a date on one example
    message: list[dict[str,str]], a list containing a dictionary with the prompt, structured as :
                                  {'role': 'user', 'content': 'prompt'}
    model: the model used for generation
    tokenizer: the model's tokenizer
    """
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    #max tokens = 13, since this is the amount of tokens that form the answer in the desired {'preicted_date' : 'DD/MM/YYYY'} format.
    outputs = model.generate(input_ids = inputs, max_new_tokens = 13, use_cache = True,)
    outputs = outputs[:,len(inputs[0]):]
    answer_only = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return answer_only[0]


def label_dataframe(df, model_checkpoint, return_formatted_date = True):
    """
    Takes a dataframe and adds a new column 'predicted_date' witht the model's prediction

    df: pd.DataFrame, a dataframe that contains at least a column called 'text' that contains the text of a document.
    model_checkpoint : str, model checkpoint to use
    return_formatted_date: bool,  if True, formatting will be applied to return only the date in DD/MM/YYYY format.
                                  if False, returns the raw output of the model
    """
    model, tokenizer = load_lora_model_inference(model_checkpoint)


    df['predicted_date'] = df['text'].apply(lambda x : predict_date(format_question(x), model, tokenizer))
    if return_formatted_date:
        df['predicted_date'] = df['predicted_date'].apply(format_predicted_date)

    return df



def main():
    parser = argparse.ArgumentParser(description="Predict dates for a dataset")

    parser.add_argument("--input-path", type=str, help="Path to file that you want to label. Can be .csv or .pkl")
    parser.add_argument("--output-path", type=str, help="Output path and filename. Can be .csv or .pkl")
    parser.add_argument("--model-checkpoint", type=str, default='zmilczarek/llama3_8b-finetuned-nlp_industry-adapters', help="Checkpoint of the model you will use")
    parser.add_argument("--gold-labels", type=str, default='none', help="The name of the column with gold labels. If passed, accuracy score will be computed")


    args = parser.parse_args()


    print(f'Input path: {args.input_path}')
    print(f'Output path: {args.output_path}')
    print(f"Checkpoint : {args.model_checkpoint}")

    in_file_type = args.input_path.split('.')[-1]
    assert in_file_type in ('csv', 'pkl'), 'Incorrect filetype passed into --input-path. Has to be either a .csv or .pkl.'

    out_file_type = args.output_path.split('.')[-1]
    assert out_file_type in ('csv', 'pkl'), 'Incorrect filetype passed into --output-path. Has to be either a .csv or .pkl.'

    if in_file_type == 'csv':
        df = pd.read_csv(args.input_path)
    elif in_file_type=='pkl':
        df = pd.read_pickle(args.input_path)

    print('Loaded the dataframe\n\n')

    df = label_dataframe(df, args.model_checkpoint)
    print('\n\nGenerated labels')

    if out_file_type == 'csv':
        df.to_csv(args.output_path, index=False)
    elif out_file_type=='pkl':
        df.to_pickle(args.output_path)

    if args.gold_labels !='':
        assert args.gold_labels in df.columns, "The gold labels column that you passed is not in the dataframe"
        acc_strict = (df['predicted_date'] == df[args.gold_labels ]).mean()
        acc_year_month = (df['predicted_date'].str[-7:] == df[args.gold_labels ].str[-7:]).mean()
        acc_year = (df['predicted_date'].str[-4:] == df[args.gold_labels ].str[-4:]).mean()
        print(f"\nAccuracy on exact date matches : {acc_strict*100:.2f}")
        print(f"Accuracy on month and year matches : {acc_year_month*100:.2f}")
        print(f"Accuracy on year matches : {acc_year*100:.2f}")



if __name__=='__main__':
    main()
