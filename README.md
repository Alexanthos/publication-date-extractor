# Publication Date Extractor

University project by Hanna Brinkmann, Zofia Milczarek, Alexandre Nechab, Joanna Radola

The aim of this project is to propose a pipeline that determines the publication date of a given document. 

A full description of the project is available in the extract_publication_date_documentation.pdf file.

## Approach
We use a Llama 3.2-Instruct 8B Model in 4bit quantization that was fine-tuned on the date extraction task to determine the publication date. The model output has consistently the following format:
~~~
{"publication date": "DD/MM/YYYY"}
~~~
For our test data set we computed the accuracy for perfect matches, month and year matches and only year matches: 

|Match              |Accuracy|
|:------------------|-------:|
|Day, Month and year|71%     |
| Month and Year    |  82%   |
| Year              | 93%    |

<br>
Before settling on this approach, we tried different, non fine-tuned models. The code can still be found above.

## Data

Our corpus consisted of 500 official documents in French, created by cities, municipalities and courts. For each document we manually annotated the publication date to have a gold standard. As not all documents were accessible, we excluded those with invalid URL. We then performed a 70/30 split in order to obtain a train and a test set. The model was fine-tuned with the first and last 3000 characters of each document in the train set. 

## How to use

In order to use the pipeline with our data, the data has to be in the right format. That can be done by running the following command. Only input and output path are mandatory arguments and possible output types are csv or pickle (pkl) files. <br>   
~~~
$ python preprocessing_finetune.py \
--input-path <original_dataset_path>
--output-path <path_for_finetuned_model> \
--out-type csv \
--length 3000 \
--no-split
~~~

Once the data is in the right format, the inference script can be called by: <br>
~~~
$ python finetuned_llama_inference.py \ 
  --input-path <dataset_path>
  --output-path <path_for_results> \
  --model-checkpoint <model_checkpoint> \
  --gold-labels <column_name_of_gold_lables>
~~~

The predicted dates will be returned in a new column in the passed dataframe. If the --gold-labels flag is passed, the script also performs evaluation and returns accuracy for perfect matches, for matches in month and year and for year matches.
