import pickle
import json, os, math
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim 
import numpy as np
import pandas as pd
BERTmodel_names=['paraphrase-multilingual-MiniLM-L12-v2','medmediani/Arabic-KW-Mdel','Ezzaldin-97/STS-Arabert','distiluse-base-multilingual-cased-v1','sentence-transformers/LaBSE']
# data_path="mydata/BASE8RGPH24V1_all_12_12_ID"
# data_path="mydata/BASE8RGPH24V4_all_18_12_ID"
data_path="mydata/Base_RGPH24V6_all_ID"
print('################################## actions run using', data_path )

df=pd.read_excel(data_path +'.xlsx')
dict_path=data_path+'unique_values_dict.json'

if os.path.exists(dict_path):
# Reading the dictionary from the JSON file
    with open(dict_path, 'r', encoding='utf-8') as file:
        unique_values_dict = json.load(file)
else:
    
## create dictionary
    unique_values_dict = {k[1]: (k[0][0],k[0][1],v,"situation") for k, v in dict(dict(zip(zip(zip(df["Situation_ID"], df["Module_ID"]), df['Situation ']),df["Situation_ID"]))).items() if (pd.notna(v) and v!='')}
    ## add tags to comparaison dict 
    unique_values_dict.update({k[1]: (k[0][0],k[0][1],v,"tags") for k, v in dict(dict(zip(zip(zip(df["Tags_ID"], df["Module_ID"]), df['Tags']),df["Situation_ID"]))).items() if (pd.notna(v) and v!='')})
    unique_values_dict.update({k[1]: (k[0][0],k[0][1],v,'situation Tags') for k, v in dict(dict(zip(zip(zip(df["tags_sit_ID"], df["Module_ID"]), df['situation Tags']),df["Situation_ID"]))).items() if (pd.notna(v) and v!='')})
    ## add section to comparaison dict
    # unique_values_dict.update({k[1]: (k[0][0],k[0][1],v,"section") for k, v in dict(dict(zip(zip(zip(df["Section_ID"], df["Module_ID"]), df['Section']),df["Situation_ID"]))).items() if (pd.notna(v) and v!='')})
    ## add question to comparaison
    unique_values_dict.update({k[1]: (k[0][0],k[0][1],v,"question") for k, v in dict(dict(zip(zip(zip(df["Question_ID"], df["Module_ID"]), df["Question AI"]),df["Situation_ID"]))).items() if (pd.notna(v) and v!='')})
    unique_values_dict = {key: value for key, value in unique_values_dict.items() if not isinstance(key, float) or not math.isnan(key)}
 
  # Writing the dictionary to a JSON file
    with open(dict_path, 'w', encoding='utf-8') as file:
        json.dump(unique_values_dict, file, ensure_ascii=False, indent=2)


BERTmodel_name=BERTmodel_names[0]
situations_list = list(unique_values_dict.keys())
# BERTmodel_name=BERTmodel_names[-1]
BERT_model=SentenceTransformer(BERTmodel_name )
pkl_path=data_path+BERTmodel_name.split('/')[0]+'situations_embeddings.pkl'

print('################# using MODEL:', BERTmodel_name)
### initialize weights

##new
# Handle BERT model embeddings
if os.path.exists(pkl_path):
    # Load sentences & embeddings from disk
    with open(pkl_path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        situations = stored_data['situations']
        BERT_weights = stored_data['BERT_weights']
    print("BERT model found")
else:
    # Encode using BERT model and save to disk
    BERT_weights = BERT_model.encode(situations_list, convert_to_tensor=True, show_progress_bar=False)
    print("BERT model fine-tuned")
    with open(pkl_path, "wb") as fOut:
        pickle.dump({'situations': unique_values_dict, 'BERT_weights': BERT_weights}, 
                     fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("BERT model saved")
print("#################################First load script ended #####################")

# Function to load data from the file
def load_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['definitions_dict'], data['module_titles'],data['choose_qst_variations'],data['definitions_dict_old'],data["other_qst_variations"]

## new
def provide_recommendations(user_input,THRESH, n, unique_values_dict,BERT_weights):
    input_weight=BERT_model.encode(user_input, show_progress_bar = True,convert_to_tensor=True)
    cosine_scores = pytorch_cos_sim(input_weight, BERT_weights)
    cosine_scores = cosine_scores.cpu().numpy()
    # Assuming cosine_scores is a 2D numpy array
    sorted_indices = np.argsort(-cosine_scores[0])  # Get indices of sorted scores in descending order
    filtered_indices = sorted_indices[cosine_scores[0][sorted_indices] >= THRESH]  # Filter based on threshold
    # Create a list of dictionaries for each situation
    ordred_situations_IDs = []
    for i in filtered_indices:
        situation_text = situations_list[i]
        unique_value = unique_values_dict[situation_text]
        ordred_situations_IDs.append({
            'input_text': user_input,
            'similar_text': situation_text,
            'element_ID': unique_value[0],
            'module_ID': unique_value[1],
            'situation_ID': unique_value[2],
            'category': unique_value[3],
            'similarity': cosine_scores[0][i]
        })
    ## process data before saving
    df_resp=add_resp_ids(df,pd.DataFrame(ordred_situations_IDs[:n]))
    df_with_qst=add_qst_ids(df,df_resp)
    # df_with_qst.to_excel(path,index=False)
    # pd.DataFrame(ordred_situations_IDs[:n]).to_excel(path,index=False)##TODO : change to json , and try to use trackers 
    # return(pd.DataFrame(ordred_situations_IDs[:n]))
    return(df_with_qst)

##new 
def add_qst_ids(df, df_rslt):
    # Create a mapping from response_ID to Question_ID
    response_to_question = df.set_index('response_ID')['Question_ID'].to_dict()
    # Function to get related question IDs
    def get_related_questions(related_responses):
        return [response_to_question.get(int(resp), []) for resp in related_responses]
    # Apply the function to the 'ralated_responses' column
    df_rslt['ralated_questions'] = df_rslt['ralated_responses'].apply(get_related_questions)
    return df_rslt



##old
def add_resp_ids(df, df_rslt):
    ids=[]
    for index, row in df_rslt.iterrows():

        if row['category']=='question':
            ids.append( df[df['Question_ID']==row["element_ID"]].response_ID.unique())
        if row['category']=='tags':
            ids.append( df[df['Tags_ID']==row["element_ID"]].response_ID.unique())
        if row['category']=='situation Tags':
            ids.append( df[df['tags_sit_ID']==row["element_ID"]].response_ID.unique())
        if row['category']=='situation':
            ids.append(df[df['Situation_ID']==row["element_ID"]].response_ID.unique())
        if row['category']=='section':
            ids.append(df[df['Section_ID']==row["element_ID"]].response_ID.unique())
    df_rslt['ralated_responses']=ids
    return df_rslt

##new !!!!!!!!!!!!! not working
# def add_resp_ids(df, df_rslt):
#     # Pre-filter DataFrames or Series for each category
#     question_ids = df.set_index('Question_ID')['response_ID']
#     tags_ids = df.set_index('Tags_ID')['response_ID']
#     tags_sit_ids = df.set_index('tags_sit_ID')['response_ID']
#     situation_ids = df.set_index('Situation_ID')['response_ID']
#     # section_ids = df.set_index('Section_ID')['response_ID']
#     # Function to get related response IDs based on category
#     def get_related_responses(row):
#         if row['category'] == 'question':
#             return question_ids.get(row["element_ID"], []).unique()
#         if row['category'] == 'tags':
#             return tags_ids.get(row["element_ID"], []).unique()
#         if row['category'] == 'situation Tags':
#             return tags_sit_ids.get(row["element_ID"], []).unique()
#         if row['category'] == 'situation':
#             return situation_ids.get(row["element_ID"], []).unique()
#         # if row['category'] == 'section':
#             # return section_ids.get(row["element_ID"], []).unique()
#         return []
#     # Apply the function to each row of df_rslt
#     df_rslt['ralated_responses'] = df_rslt.apply(get_related_responses, axis=1)
#     return df_rslt



##new
def module_recommendations(df_rslt, n=3):
    # Get the first n unique module IDs
    module_ids = df_rslt['module_ID'].dropna().unique()[:n]
    # Retrieve module names for these IDs
    module_names = df[df.Module_ID.isin(module_ids)]['module'].drop_duplicates().tolist()
    return module_ids.tolist(), module_names


##new
def situation_recommendations(df_rslt, module_id, n=3, nan_id=5):
    # Filter the DataFrame and get unique situation IDs, excluding nan_id
    filtered_situation_ids = df_rslt[(df_rslt['module_ID'] == int(module_id)) & (df_rslt['situation_ID'] != nan_id)]['situation_ID'].unique()
    # Limit the number of situation IDs to n
    output_ids = filtered_situation_ids[:n].tolist()
    # Fetch situation names for these IDs
    # situation_names = df[df['Situation_ID'].isin(output_ids)]['Situation '].drop_duplicates().tolist() ### order problem
    situation_names=[df[df.Situation_ID==output_id]["Situation "].unique().tolist()[0] for output_id in output_ids]
    return output_ids, situation_names

##new
def question_recommendations(df_rslt_with_qst, situation_ID, n=3):
    # Check the first row's conditions
    first_row = df_rslt_with_qst.iloc[0]
    if first_row['category'] == "question" and first_row['similarity'] > 0.8:
        questions = [first_row['ralated_responses']]
    else:
        questions = []
    # Extend with questions related to the situation ID
    questions.extend(df_rslt_with_qst[df_rslt_with_qst['situation_ID'] == situation_ID]['ralated_responses'])
    # Flatten the list of lists
    flat_questions = [item for sublist in questions for item in sublist]
    # Precompute response to question ID mapping
    response_to_question = df.set_index('response_ID')['Question_ID'].to_dict()
    # Precompute question ID to question name mapping
    question_to_name = df.set_index('Question_ID')['Question AI'].to_dict()
    # Select questions
    selected_questions, question_names, extra_questions, extra_question_names = [], [], [], []
    for response_id in flat_questions:
        question_id = response_to_question.get(int(response_id))
        if question_id and question_id not in selected_questions and len(selected_questions) < n:
            selected_questions.append(question_id)
            question_names.append(question_to_name.get(question_id))
        elif question_id and question_id not in selected_questions and question_id not in extra_questions:
            extra_questions.append(question_id)
            extra_question_names.append(question_to_name.get(question_id))
    return selected_questions, question_names, extra_questions, extra_question_names

##new
def get_responses(question_id):
    # Use loc for efficient row selection and drop_duplicates for unique responses
    response = df.loc[df.Question_ID == question_id, 'Réponse  Quasi-finale'].drop_duplicates().tolist()
    return response
