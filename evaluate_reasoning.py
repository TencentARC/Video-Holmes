import json
import os
from openai import OpenAI
from tqdm import tqdm
import time
import re
import argparse

wrong_prompt = '''
You are an expert in logic. Your task is to analyze the cause of errors in a multimodal large model's responses to video reasoning tasks. The requirements are as follows:

I will provide you with a plot description of a short reasoning film, a question, options, the correct answer, and an explanation of the correct answer.Then I will provide you with the [thought process] of a multimodal large model and its incorrect answer. Your task is to compare the multimodal large model's thought process with the correct explanation and the film plot. Determine the most critical cause of the error from the following: 

(1) VPE (Visual Perception Error): The model extracted incorrect visual information for analysis, leading to the wrong answer. (For example, the video shows a man robbing with a knife, but the model perceives it as a man robbing with a gun.) 
(2) VOE (Visual Omission Error): The model failed to extract key visual information (such as key objects and events), leading to the wrong answer. 
(3) RE (Reasoning Error): The model made an error in the reasoning process, misinterpreting the implications of the visual information and incorrectly judging the relationships between multiple pieces of visual information. 
(4) TRAW (Think Right Answer Wrong): The model's thought process is generally consistent with the answer explanation, but it chose the wrong option when answering.

Your response format: Please provide the error abbreviation within <Type></Type>. Then give your brief judgment reason within <Reason></Reason>.
'''

right_prompt = '''
You are an expert in logic. Your task is to analyze the thought process of a multimodal large model in responding to video reasoning tasks. The requirements are as follows:

I will provide you with a [plot description] of a short reasoning film, a question, options, the correct answer, and an explanation of the correct answer. Then I will provide you with the [model's thought process] and its correct answer. Your task is to compare the multimodal large model's thought process with the correct explanation and the film plot. Determine which type the response belongs to:
(1) TWAR (Think Wrong Answer Right): The model's thought process shows a significant deviation from the answer explanation, using incorrect information to arrive at the correct conclusion. 
(2) TRAR (Think Right Answer Right): The model's thought process is generally consistent with the answer explanation (allowing for minor deviations).

Your response format: Please provide the response type abbreviation within <Type></Type>. Then give your brief judgment reason within <Reason></Reason>.
'''




def analysis(client, segment_str, question, options, correct_answer, predict_answer, correct, thinking, explanation):
    sys_prompt = f'''
        Here are the information:
        [Film Description]：{segment_str}
        [Question]：{question}
        [Options]：{options}
        [Correct Answer]：{correct_answer}
        [Explanation]：{explanation}
        [Model's Answer]：{predict_answer}
        [Model's Thinking Process]：{thinking}
    '''

    if correct:
        prompt = right_prompt + sys_prompt
    else:
        prompt = wrong_prompt + sys_prompt

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        output = response.choices[0].message.content
    except:
        print("Error occurred while calling the API")

    print(output)
    output_type = "Nan"
    output_reason = "Nan"

    type_pattern = r'<Type>\s*(.*?)\s*</Type>'
    match = re.search(type_pattern, output, re.DOTALL)
    if match:
        output_type = match.group(1).strip()

    reason_pattern = r'<Reason>\s*(.*?)\s*</Reason>'
    match = re.search(reason_pattern, output, re.DOTALL)
    if match:
        output_reason = match.group(1).strip()

    return output_type, output_reason



def evaluate(annotation_path, results_path, model_name, client):
    json_file = os.path.join(results_path, f"Results-{model_name}.json")
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for item in tqdm(data):
        explanation = item.get('Explanation')
        question = item.get('Question')
        options = item.get('Options')
        options = ', '.join([f"{key}: {value}" for key, value in options.items()])
        correct_answer = item.get('GT')
        predict_answer = item.get('Predicr Answer')
        correct = item.get('Correct')
        thinking = item.get('Thinking')
        video_id = item.get('video ID')
        timeline_json = os.path.join(annotation_path, f"{video_id}.json")

        with open(timeline_json, 'r', encoding='utf-8') as file:
            time_data = json.load(file)
            segment_description = time_data["SegmentDescription"]
            segment_str = json.dumps(segment_description, ensure_ascii=False)

        if thinking == "":
            item['Answer Type'] = "Nan"
            item['Reasoning Process Analysis'] = "Nan"
        else:
            item['Answer Type'], item['Answer Type Reason'] = analysis(client, segment_str, question, options, correct_answer, predict_answer, correct, thinking, explanation)
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model with specified name")
    parser.add_argument('--model_name', type=str, default="Qwen2.5-VL-7B", help="Name of the model to evaluate")
    parser.add_argument('--api_key', type=str, help="API key for DeepSeek")
    args = parser.parse_args()

    model_name = args.model_name
    annotation_path = 'Benchmark/annotations'
    results_path = 'Results/'
    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")

    evaluate(annotation_path, results_path, model_name, client)

