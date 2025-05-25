import json
import time
import os
import glob
from tqdm import tqdm
import re
from openai import OpenAI
import argparse


base_prompt = '''
You are a professional logician and detective story enthusiast. I will provide you with the plot information of a short detective film. Your task is to create some reasoning multiple-choice questions for viewers who have watched this short detective film to test whether they truly understood it.

Requirements:
I will provide the following information:
(1) Segmental plot description: chronological plot description
(2) Key character relationships and cause positioning: the relationships between key characters in the film and the information that can be used to infer their relationships.
(3) Reasoning shots: key hint shots in the plot, plot twists, and climaxes.
(4) Supernatural elements: supernatural elements and rules appearing in the video.
(5) Main idea of the video: the possible main idea the video wants to express.

You need to create at least one multiple-choice question from each of the following seven types of reasoning (according to the specific requirements):


Type 1: Social Reasoning (SR)
Task description: infer the implicit social relationship network between characters through clothing style matching, group interaction topology analysis, including identity association across time spans (such as the same character in youth and old age).
Number of questions: Create one question for each combination based on the given key character relationships. If the [segmental plot description] provided includes the same character across time spans, create a question based on this.
Example 1: What is the relationship between the man at the beginning of the video and the skeleton at the end? (Answer: A. The same person)

Type 2: Intention and Motive Chaining (IMC)
Task description: infer the underlying psychological state and predict non-explicit behavioral intentions by observing characters' actions, expressions, or environmental clues. The question format must strictly follow the example: [Character] at [time] performed [very brief, non-key detail action description] with what intention/psychological state?
\end{tcolorbox}
\begin{tcolorbox}[breakable, colback=gray!5!white, colframe=gray!75!black, 
title=Question Generation Prompt (Continue), boxrule=0.5mm, width=\textwidth, arc=3mm, auto outer arc]
Number of questions: 1-2 questions
Example 1: What is the most likely psychological state of the man in black smoking at 00:03 in the video? (Answer: C. Anxiety about preparing for a crime)

Type 3: Temporal Causal Inference (TCI)
Task description: infer causal mechanisms between non-continuous temporal events through camera language and multimodal clues.
Number of questions: 1-2 questions
Example 1: Why did the man in the shirt die? (Answer: F. Overuse of superpowers)

Type 4: Timeline Analysis (TA)
Task description: reorganize video clips in chronological order. You need to provide 5 key events and ask to restore the timeline.
Number of questions: 1 question
Example 1: 1. The man in black wakes up 2. The man in black is attacked 3. The man in black goes out 4. The man in black is sleeping 5. The man in black meets the woman

Type 5: Multimodal Hint Reasoning (MHR)
Task description: analyze the visual and auditory hints set by the director: camera movement semantics, object position changes, sound and picture metaphors. Ask about specific camera transitions, the appearance, movement, or change of objects.
Number of questions: Create one question for each given hint shot based on the given [reasoning conclusion].
Example 1: The correct interpretation of the dancing scene between the man and woman in the video is (Answer: A. The man's own fantasy)

Type 6: Physical Anomaly Reasoning (PAR)
Task description: identify scenes in the video that do not conform to reality logic (such as magic, sci-fi elements). The first question asks about the rules of supernatural elements, and the second question asks about their implied meaning.
Number of questions: (2n questions) Create questions based on the given supernatural elements. If none are provided, skip this type.

Type 7: Core Theme Inference (CTI)
Task description: infer the core theme or deep meaning through the plot, dialogue, or symbols in the video. (The given main idea of the video is a subjective conclusion and may not be entirely correct)

The questions you propose need to be understood and carefully reasoned to answer. Each question needs to provide 6 options, ensuring the correct option is objectively unique (distractor options cannot express a similar meaning to the correct option, they must be clearly distinguished), and the distractor options should overlap with the correct option as much as possible (e.g., A. Because the man forgot to bring his phone B. Because the man forgot to bring his tablet C. Because the man forgot to bring his wallet). You need to provide the question type, question, answer, and explanation. Key point: The question stem cannot reveal character psychology, factual information, hint clues, and hint conclusions, viewers must find them themselves. For example: Bad question: "What does the skeleton at the end of the video imply?" - Good question: "What does the skeleton at the end of the video imply?"; Bad question: "What is the psychological state of the man when he imagines the future in the lighting change scene?" - Good question: "What is the psychological state of the man in the lighting change scene?"

Answer format example:
For each question, the 1st line: [Question Type]: CTI The 2nd line: [Question]: Your question; The 3rd-8th lines: A-F options; The 9th line: [Answer]: F; The 10th line: [Explanation]: Your explanation.
'''


def generate_questions(client, output_path, description_path) :

    os.makedirs(output_path, exist_ok=True)
    anno_files = glob.glob(os.path.join(description_path, '*.json'))

    for anno_file in anno_files:
        output_json_path = os.path.join(output_path, os.path.basename(anno_file))

        if os.path.exists(output_json_path):
            print(f'Exist {output_json_path}')
            # continue

        print(f"Reading file: {anno_file}")
        with open(anno_file, 'r', encoding='utf-8') as file:
            content = file.read()

           
          
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": base_prompt + f"The key information of the film are: \n{content}"},
            ],
            stream=False
        )
        output = response.choices[0].message.content

        print(output)
        questions = output.strip().split("\n\n")
        parsed_questions = []
        video_name = anno_file.split('/')[-1].replace('.json','')
        ok_q = 1
        for question in questions:
            lines = question.strip().split("\n")
            question_dict = {"video ID" : video_name, "Question ID" : 1, "Question Type" : "", "Question" : "", "Options" : {}, "Answer" : "", "Explanation" : ""}
            
            for index, line in enumerate(lines):
                if line.startswith("[Question Type"):
                    question_dict["Question Type"] = line.replace(': ',':').split(":", 1)[1].strip().replace(' ','')
                elif line.startswith("[Question]:"):
                    question_dict["Question"] = line.replace(': ',':').split(":", 1)[1].strip()
                    for i in range(10):
                        if not lines[index+1+i].startswith('A'):
                            question_dict["Question"] = question_dict["Question"] + "\n" + lines[index+1+i]
                        else:
                            break

                elif line.startswith("[Answer]:"):
                    question_dict["Answer"] = line.replace(': ',':').split(":", 1)[1].strip()

                elif line.startswith("[Explanation]:"):
                    question_dict["Explanation"] = line.replace(': ',':').split(":", 1)[1].strip()

                elif 'A.' in line and 'B.' in line and 'C.' in line:
                    matches = re.findall(r'([A-Z])\. ([^A-Z]+)', line)
                    for key, value in matches:
                        question_dict["Options"][key] = value.strip().replace(' ','')

                else:
                    if 'A.' in line:
                        question_dict["Options"]["A"] = line.strip().replace('A. ','')
                    if 'B.' in line:
                        question_dict["Options"]["B"] = line.strip().replace('B. ','')
                    if 'C.' in line:
                        question_dict["Options"]["C"] = line.strip().replace('C. ','')
                    if 'D.' in line:
                        question_dict["Options"]["D"] = line.strip().replace('D. ','')
                    if 'E.' in line:
                        question_dict["Options"]["E"] = line.strip().replace('E. ','')
                    if 'F.' in line:
                        question_dict["Options"]["F"] = line.strip().replace('F. ','')

            if question_dict["Question Type"] == "" or question_dict["Question"] == "":
                continue
            else:
                question_dict["Question ID"] = ok_q
                ok_q += 1

            parsed_questions.append(question_dict)
        json_output = json.dumps(parsed_questions, ensure_ascii=False, indent=4)
        
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_output)




    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model with specified name")
    parser.add_argument('--api_key', type=str, help="API key for DeepSeek")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    output_path = 'example_questions/'
    description_path = 'example_annotations/'

    generate_questions(client, output_path, description_path) 