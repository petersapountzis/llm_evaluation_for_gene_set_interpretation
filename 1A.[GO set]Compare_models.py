#%% md
# ## Set up model comparison
# 
# **gpt model**
# 
# gpt-4-1106-preview
# 
# **local model on server**
# 
# available models:
# 
# | NAME           | ID           | SIZE   |
# |----------------|--------------|--------|
# | llama2:70b     | c3a7af098300 | 38 GB  |
# | llama2:7b      | fe938a131f40 | 3.8 GB |
# | llama2:latest  | fe938a131f40 | 3.8 GB |
# | mistral:7b     | 4d9f4b269c33 | 4.1 GB |
# | mixtral:latest | 99a9202f8a7a | 26 GB  |
# | mixtral:instruct| 7708c059a8bb | 26 GB  |	
# 
# 
# 
# **API for calling Google Gemini pro**
# 
# GO TO: https://makersuite.google.com/app/apikey to get the apikey for gemini pro
# 
# export GOOGLEAI_KEY = xxxx
# 
# model = 'gemini-pro'
#%%
import pandas as pd
import numpy as np
import json 
from utils.openai_query import openai_chat
from utils.prompt_factory import make_user_prompt_with_score
from utils.server_model_query import server_model_chat
from utils.llm_analysis_utils import process_analysis, save_progress
from utils.genai_query import query_genai_model
from tqdm import tqdm
import constant
import openai
import os
import logging
import re
%load_ext autoreload

%autoreload 2

#%% md
# **Example for running in the jupyter notebook**
#%%
## load variables
initialize = True # if True, then initialize the input table with llm names, analysis and score to None 
# Replace with your actual values
config_file = './jsonFiles/toyexample_gpt35.json'  # replace with your actual config file 
input_file = './data/GO_term_analysis/100_selected_go_contaminated.csv' # replace with your actual input file
input_sep = ','  # replace with the separator
set_index = 'GO'  # replace with your column name that you want to set as index or None
gene_column = 'Genes'  # replace with your actual column name for the gene list
gene_sep = ' '  # replace with your actual separator
gene_features = None  # replace with your path to the gene features or None if you don't want to include in the prompt
direct = False # if True, then the prompt will be a direct sentence asking for a name and analysis from the gene set, otherwise default or customized prompt
out_file = 'data/GO_term_analysis/model_compare/PETER_COMPARE_TEST'  # replace with your actual output file name

customized_prompt = False # if True, then the prompt will be the custom prompt, if False, then the prompt will use default

# load the config file
with open(config_file) as json_file:
    config = json.load(json_file)

if customized_prompt:
    # make sure the file exist 
    if os.path.isfile(config['CUSTOM_PROMPT_FILE']):
        with open(config['CUSTOM_PROMPT_FILE'], 'r') as f: # replace with your actual customized prompt file
            customized_prompt = f.read()
            assert len(customized_prompt) > 1, "Customized prompt is empty"
    else:
        print("Customized prompt file does not exist")
        customized_prompt = None
else:
    customized_prompt = None

# Load OpenAI key, context, and model used
openai.api_key = os.environ["OPENAI_API_KEY"]

context = config['CONTEXT']
model = config['MODEL']
temperature = config['TEMP']
max_tokens = config['MAX_TOKENS']
if model.startswith('gpt'):
    rate_per_token = config['RATE_PER_TOKEN']
    DOLLAR_LIMIT = config['DOLLAR_LIMIT']
LOG_FILE = config['LOG_NAME']+'_.log'

SEED = constant.SEED
column_prefix = model.split('-')[0]
#%%
# handle the logger so it create a new one for each model run
def get_logger(filename):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def main(df):
    analysis_dict  = {}

    logger = get_logger(f'{out_file}.log')

    i = 0 #used for track progress and saving the file
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        #only process None rows 
        if pd.notna(row[f'{column_prefix} Analysis']):
            continue
        
        gene_data = row[gene_column]
        # if gene_data is not a string, then skip
        if type(gene_data) != str:
            
            logger.warning(f'Gene set {idx} is not a string, skipping')
            continue
        genes = gene_data.split(gene_sep)
        
        if len(genes) >1000:
            logger.warning(f'Gene set {idx} is too big, skipping')
            continue

        try:
            prompt = make_user_prompt_with_score(genes)
            # print(prompt)
            finger_print = None
            if model.startswith('gpt'):
                print("Accessing OpenAI API")
                analysis, finger_print = openai_chat(context, prompt, model, temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT, SEED)
            elif model.startswith('gemini'):
                print("Using Google Gemini API")
                analysis, error_message = query_genai_model(f"{context}\n{prompt}", model, temperature, max_tokens, LOG_FILE) 
            else:
                print("Using server model")
                analysis, error_message= server_model_chat(context, prompt, model, temperature, max_tokens,LOG_FILE, SEED)

            
            if analysis:
                # print(analysis)
                llm_name, llm_score, llm_analysis = process_analysis(analysis)
                # clean up the score and return float
                try:
                    llm_score_value =  float(re.sub("[^0-9.-]", "", llm_score))
                except ValueError:
                    llm_score_value = llm_score
            
                
                df.loc[idx, f'{column_prefix} Name'] = llm_name
                df.loc[idx, f'{column_prefix} Analysis'] = llm_analysis
                df.loc[idx, f'{column_prefix} Score'] = llm_score_value
                
                analysis_dict[f'{idx}_{column_prefix}'] = analysis
                # Log success with fingerprint
                logger.info(f'Success for {idx} {column_prefix}.')
                if finger_print:
                    logger.info(f'GPT_Fingerprint for {idx}: {finger_print}')
                    
            else:
                logger.error(f'Error for query gene set {idx}: {error_message}')

        except Exception as e:
            logger.error(f'Error for {idx}: {e}')
            continue
        i += 1
        if i % 10 == 0:
            # bin scores into no score, low score, medium score, high score
            bins = [-np.inf, 0, 0.79, 0.86, np.inf] # 0 is no score (name not assigned), between 0 to 0.79 is low score, between 0.8 to 0.86 is medium score, above 0.86 is high score
            labels = ['Name not assigned', 'Low Score', 'Medium Score', 'High Score']  # Define the corresponding labels
            
            df[f'{column_prefix} Score bins'] = pd.cut(df[f'{column_prefix} Score'], bins=bins, labels=labels)
                
            save_progress(df, analysis_dict, out_file)
            # df.to_csv(f'{out_file}.tsv', sep='\t', index=True)
            print(f"Saved progress for {i} genesets")
    # save the final file
    save_progress(df, analysis_dict, out_file)
    
#%%
import os 
from glob import glob


initialize = True 
input_file = 'data/GO_term_analysis/toy_example_w_contaminated.csv'
input_sep = constant.GO_FILE_SEP
set_index = constant.GO_INDEX_COL  
gene_column = constant.GO_GENE_COL 
gene_sep = ' '

## create a param file 
configs = glob('./jsonFiles/toyexample_*.json')
params = []
for conf_file in configs:
    model_names = '_'.join(conf_file.split('/')[-1].split('.')[0].split('_')[1:])
    # print(model_names)
    out_file = f'data/GO_term_analysis/LLM_processed_toy_example_w_contamination_{model_names}'  
    param = f"--config {conf_file} \
        --initialize \
        --input {input_file} \
        --input_sep  '{input_sep}'\
        --set_index {set_index} \
        --gene_column {gene_column}\
        --gene_sep '{gene_sep}' \
        --start 0 \
        --end 10 \
        --output_file {out_file}"
    print(param)
    params.append(param)

with open('toy_example_params.txt', 'w') as f:
    for p in params:
        f.write(p+'\n')
#%%
#Define your own loop for running the pipeline
## 12-18-2023: this loop is for run the default gene set and the contaminated gene sets 
## can modify this loop for different models or only run on default gene set

##12-27-23: edited the prompt 

##01-26-2023: test with bin scores

if __name__ == "__main__":
    
    df = pd.read_csv(input_file, sep=input_sep, index_col=set_index)
    
    if 'gpt' in model:
        name_fix = '_'.join(model.split('-')[:2])
    else:
        name_fix = model.replace(':', '_')
    column_prefix = name_fix + '_default'
    print(column_prefix)
    if initialize:
        # initialize the input file with llm names, analysis and score to None
        df[f'{column_prefix} Name'] = None
        df[f'{column_prefix} Analysis'] = None
        df[f'{column_prefix} Score'] = -np.inf
    main(df)  ## run with the real set 
    
    ## run the pipeline for contaiminated gene sets 
    contaminated_columns = [col for col in df.columns if col.endswith('contaminated_Genes')]
    # print(contaminated_columns)
    for col in contaminated_columns:
        gene_column = col ## Note need to change the gene_column to the contaminated column
        contam_prefix = '_'.join(col.split('_')[0:2])
        
        column_prefix = name_fix + '_' +contam_prefix
        print(column_prefix)

        if initialize:
            # initialize the input file with llm names, analysis and score to None
            df[f'{column_prefix} Name'] = None
            df[f'{column_prefix} Analysis'] = None
            df[f'{column_prefix} Score'] = -np.inf
        main(df)
    df.head()

#%%
# check if there is any None in the analysis column, then rerun the pipeline

initialize = False 

SEED = 42
model_options = ['gemini-pro','mistral:7b', 'mixtral:latest', 'llama2:7b', 'llama2:70b']
# model_options = ['mixtral:latest']  # llama2 7b has formatting issue, ingore 
input_sep = '\t'

if __name__ == "__main__":
    for m in model_options:
        model = m
        
        if '-' in model:
            name_fix = '_'.join(model.split('-')[:2])
        else:
            name_fix = model.replace(':', '_')
        input_file = f'data/GO_term_analysis/LLM_processed_toy_example_w_contamination_{name_fix}.tsv' # replace with your actual input file
        out_file = f'data/GO_term_analysis/LLM_processed_toy_example_w_contamination_{name_fix}'  # save to the same file name as the input file
        LOG_FILE = config['LOG_NAME']+f'_{name_fix}'+'.log'

        df = pd.read_csv(input_file, sep=input_sep, index_col=set_index)
        # print(df.head())
        column_prefix = name_fix + '_default' #this is default
        print(column_prefix)
        
        gene_column = constant.GO_GENE_COL
        print(gene_column)
        if initialize:
            # initialize the input file with llm names, analysis and score to None
            df[f'{column_prefix} Name'] = None
            df[f'{column_prefix} Analysis'] = None
            df[f'{column_prefix} Score'] = None
        main(df)  ## run with the real set 
        
        ## run the pipeline for contaiminated gene sets 
        contaminated_columns = [col for col in df.columns if col.endswith('contaminated_Genes')]
        # print(contaminated_columns)
        for col in contaminated_columns:
            gene_column = col ## Note need to change the gene_column to the contaminated column
            print(gene_column)
            contam_prefix = '_'.join(col.split('_')[0:2])
            column_prefix = name_fix + '_' +contam_prefix
            print(column_prefix)

            if initialize:
                # initialize the input file with llm names, analysis and score to None
                df[f'{column_prefix} Name'] = None
                df[f'{column_prefix} Analysis'] = None
                df[f'{column_prefix} Score'] = None
            main(df)
            
print("Done")
#%% md
# ## Run for the 100 sets
#%%
## set up parameters for running the pipeline for every 50 rows
import os 
from glob import glob
# Define start, step, and end values
start = 0
step = 50
end = 100

# Create a range list
range_list = list(range(start, end + step, step))

# Create tuples for each consecutive pair in the list
tuple_list = [(range_list[i], range_list[i+1]) for i in range(len(range_list)-1)]


initialize = True 
input_file = 'data/GO_term_analysis/model_comparison_terms.csv'
input_sep = constant.GO_FILE_SEP
set_index = constant.GO_INDEX_COL  
gene_column = constant.GO_GENE_COL 
gene_sep = ' '

## create a param file 
configs = glob('./jsonFiles/model_comparison_*.json')
params = []
for start, end in tuple_list:
    for conf_file in configs:
        model_names = '_'.join(conf_file.split('/')[-1].split('.')[0].split('_')[1:])
        print(model_names)
        
        out_file = f'data/GO_term_analysis/model_compare/LLM_processed_model_compare_{model_names}_{start}_{end}'  
        param = f"--config {conf_file} \
            --initialize \
            --input {input_file} \
            --input_sep  '{input_sep}'\
            --set_index {set_index} \
            --gene_column {gene_column}\
            --gene_sep '{gene_sep}' \
            --run_contaminated \
            --start {start} \
            --end {end} \
            --output_file {out_file}"
        print(param)
        params.append(param)
print('number of params: ', len(params))
    

with open('model_compare_params.txt', 'w') as f:
    for p in params:
        f.write(p+'\n')
#%% md
# ## Checkout and combine the output from the batch run 
#%%
from glob import glob
import pandas as pd
import json

processed_files = glob('data/GO_term_analysis/model_compare/LLM_processed_model_compare*.tsv')
# processed_files
# check any with None in the analysis column
for file in processed_files:
    model_names = '_'.join(file.split('/')[-1].split('.')[0].split('_')[-4:])
    
    df = pd.read_csv(file, sep='\t')
    # column names end with Analysis
    analysis_cols = [col for col in df.columns if col.endswith('Analysis')]
    for col in analysis_cols:
        if df[col].isna().sum() > 0:
            n_none = df[col].isna().sum()
            print(f'{model_names} {col} has {n_none} None in the analysis column')
        else:
            print(f'{model_names} {col} pass')
        print('-----------------------')
    

    
#%%
## combine the 0-50 and 50-100 files together
from glob import glob
import pandas as pd
import json

processed_files = glob('data/GO_term_analysis/model_compare/LLM_processed_model_compare*.tsv')

# model_names = ['mixtral_instruct']
for file in processed_files:
    model_name = '_'.join(file.split('/')[-1].split('.')[0].split('_')[-4:-2])
    model_names.append(model_name)
model_names = list(set(model_names))

for model in model_names:
    print(model)
    files = [file for file in processed_files if model in file]
    print(files)
    df = pd.concat([pd.read_csv(file, sep='\t', index_col='GO') for file in files])
    
    # add the toy example in as well 
    toy_file = f'data/GO_term_analysis/LLM_processed_toy_example_w_contamination_{model}.tsv'
    
    df = pd.concat([df, pd.read_csv(toy_file, sep='\t', index_col='GO')])
    # check any with None in the analysis column
    analysis_columns = [col for col in df.columns if col.endswith('Analysis')]
    for col in analysis_columns:
        if df[col].isna().sum() > 0:
            n_none = df[col].isna().sum()
            print(f'{model} {col} has {n_none} None in the analysis column')
    
    print(df.shape)
    df.to_csv(f'data/GO_term_analysis/model_compare/LLM_processed_model_compare_100set_{model}.tsv', sep='\t', index=True)
    print('------------saved--------------')
#%%
##check for each 100 set file, how many 'systems of unrelated proteins' are assigened to each gene set 
from glob import glob
import pandas as pd
import json

files = glob('data/GO_term_analysis/model_compare/LLM_processed_model_compare_100set*.tsv')
unnamed_dict = {}
model_names = []

for file in files:
    model_name = '_'.join(file.split('/')[-1].split('.')[0].split('_')[-2:])
    model_names.append(model_name)
    df = pd.read_csv(file, sep='\t', index_col='GO')
    name_columns = [col for col in df.columns if col.endswith('Name')]
    
    for col in name_columns:
        gene_set_type = col.split(' ')[0]
        # print(gene_set_type)
        #number of names contains 'unrelated proteins'
        n_unrelated = df[col].str.contains('unrelated proteins').sum()
        n_total = df.shape[0]
        print(f'{gene_set_type} has {n_unrelated} gene sets named with unrelated proteins, {n_unrelated/n_total*100:.2f}%')
        unnamed_dict[f'{gene_set_type}'] = {'n_unrelated': n_unrelated, 'n_named': n_total-n_unrelated}
    score_columns = [col for col in df.columns if col.endswith('Score')]
    for c in score_columns:
        gene_set_type = c.split(' ')[0]
        # print(gene_set_type)
        # number of scores are 0
        n_zero = df[c].eq(0).sum()
        n_total = df.shape[0]
        print(f'{gene_set_type} has {n_zero} gene sets with score 0, {n_zero/n_total*100:.2f}%')
        
    print('------------------')
#%%
## check the time for each model 
from glob import glob
import json
log_files = glob('./logs/model_comparison_*.log')
# print(log_files)
models = ['gpt_4', 'gemini_pro', 'mixtral_instruct','llama2_70b']
total_time_per_run = 0
for model in models:
    logs = [file for file in log_files if model in file]
    time = 0
    runs = 0
    for log in logs:
        with open(log, 'r') as f:
            data = json.load(f)
        time += data['time_taken_total']
        runs += data['runs']
        if  model == 'gpt_4':
            cost = data['dollars_spent']/data['runs']
    time_per_run = time/runs
    if model in ['mixtral_instruct','llama2_70b']:
        time_per_run = time_per_run/ 1e9 
        
    print(f'{model} takes {time_per_run :.2f} seconds per run')
    if  model == 'gpt_4':
        print(f'{model} takes {cost} dollars per run')
    total_time_per_run += time_per_run

with open ('/cellar/users/mhu/Projects/llm_evaluation_for_gene_set_interpretation/logs/toy_example_gpt35_.log', 'r') as f:
    data = json.load(f)
    time = data['time_taken_total']
    runs = data['runs']
    time_per_run = time/runs  
    total_time_per_run += time_per_run 
    cost = data['dollars_spent']/runs
print(f'gpt 3.5 takes {time_per_run:.2f} seconds per run')  
print(f'gpt 3.5 takes {cost} dollars per run')
print('average time usage: ', total_time_per_run/5)

#%%
