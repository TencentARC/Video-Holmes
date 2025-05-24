import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import json
from tqdm import tqdm
import time
import random
import csv
from collections import defaultdict
import math
import numpy as np
import re
from PIL import Image
import argparse


import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def prepare_internvl_25_8b(model_path):
    path = model_path if model_path is not None else "OpenGVLab/InternVL2_5-8B"
    
    device_map = split_model('InternVL2_5-8B')
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    return model, tokenizer, generation_config

def predict_internvl(model, tokenizer, generation_config, question_prompt, video_path):

    pixel_values, num_patches_list = load_video(video_path, num_segments=32, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + question_prompt
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                num_patches_list=num_patches_list, history=None, return_history=True)
    return response

def prepare_internvl_3(model_path):

    def split_model(model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        return device_map
    
    path = model_path if model_path is not None else "OpenGVLab/InternVL3-8B"

    device_map = split_model(path)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    return model, tokenizer, generation_config

def chat_with_multi_modal(model: str, prompt: str, video_file):
    import google.generativeai as genai 

    safety_settings={
        'HATE': 'BLOCK_NONE',
        'HARASSMENT': 'BLOCK_NONE',
        'SEXUAL': 'BLOCK_NONE',
        'DANGEROUS': 'BLOCK_NONE'
    }

    model = genai.GenerativeModel(model_name=model)
    ret = model.generate_content(
        [video_file, prompt],
        request_options={"timeout": 600},
        safety_settings=safety_settings
    )
    return ret.text

def predict_gemini(api_key, model_name, query, dst):
        import google.generativeai as genai 
        response = "WRONG"

        genai.configure(api_key=api_key)
        local_file_path = dst
        print(f"Uploading file: {local_file_path}")

        video_file = genai.upload_file(path=local_file_path)

        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(30)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            print(f'Failed to upload file: {video_file.state.name}')
            exit(-1)

        print(f"Completed upload: {video_file.uri}")
        response = chat_with_multi_modal(model_name, query, video_file)

        return response

def prepare_Qwen_Family(model_name, model_path):
    from transformers import AutoProcessor, AutoTokenizer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    if model_path:
        MODEL_PATH = model_path
    elif model_name == "Qwen2.5-VL-7B":
        MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    elif model_name == "Qwen2.5-VL-32B":
        MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"
    elif model_name == "VideoChat-R1":
        MODEL_PATH = "OpenGVLab/VideoChat-R1_7B"
    elif model_name == "Video-R1":
        MODEL_PATH = "Video-R1/Video-R1-7B"

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len = 15000,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
        stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    return llm, processor, sampling_params

def predict_Qwen_Family(llm, processor, sampling_params, question_prompt, video):
    from transformers import AutoProcessor, AutoTokenizer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    messages = [[
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video,
                },
                {"type": "text", "text": question_prompt},
            ],
        }
    ]]

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        

    mean_acc = []
    mean_mra = []
    batch_messages = messages[:]
    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        batch_messages, 
        return_video_kwargs=True,
        )

    image_idx = 0
    video_idx = 0
    llm_inputs = []

            
    for idx, prompt in enumerate(prompts):
        
        mm_type = batch_messages[idx][0]['content'][0]['type']
        sample_mm_data = {}
        sample_video_kw = {}
        if mm_type == 'image':
            sample_mm_data["image"] = image_inputs[image_idx]
            image_idx += 1
        elif mm_type == 'video':
            sample_mm_data["video"] = video_inputs[video_idx]
            for key, value in video_kwargs.items():
                sample_video_kw[key] = value[video_idx]
            video_idx += 1
                    
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": sample_mm_data,
                "mm_processor_kwargs": sample_video_kw,
            })
            

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        batch_output_text = [out.outputs[0].text for out in outputs]
        return batch_output_text[0]

def evaluate(video_path, json_file_path, output_path, model_name, model_path):

    # Prepare Model 

    if model_name == "InternVL2.5-8B":
        model, tokenizer, generation_config = prepare_internvl_25_8b(model_path)

    elif model_name == "InternVL3-8B":
        model, tokenizer, generation_config = prepare_internvl_3(model_path)

    elif "Qwen2.5-VL" in model_name or "VideoChat-R1" in model_name or "Video-R1" in model_name:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
        from vllm import LLM, SamplingParams
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, AutoTokenizer
        llm, processor, sampling_params = prepare_Qwen_Family(model_name, model_path)

    elif "gemini" in model_name:
        api_key = "YOUR GOOGLE API KEY"

    # else:
    #     model = prepare_your_model()

    os.makedirs(output_path,exist_ok=True)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    question_types = ['SR', 'IMC', 'TCI', 'TA', 'MHR', 'PAR', 'CTI']

    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    output_process = []
    json_file_output = os.path.join(output_path, f"Results-{model_name}.json")

    general_qid_dict = {}
    if os.path.exists(json_file_output):
        with open(json_file_output, "r", encoding="utf-8") as f:
            output_process = json.load(f)
            for item in output_process:
                question_id = item.get("Question ID")
                if question_id:
                    general_qid_dict[question_id] = 1

    for item in tqdm(data):
        if len(output_process) > 0:
            if item["Question ID"] in general_qid_dict.keys():
                print(item["Question ID"])
                continue

        quesion_id = item.get('Question ID')
        explanation = item.get('Explanation')
        q_type = item.get('Question Type')
        question = item.get('Question')
        options = item.get('Options')
        options = ', '.join([f"{key}: {value}" for key, value in options.items()])
        correct_answer = item.get('Answer')
        video_id = item.get('video ID')
        video = os.path.join(video_path, f"{video_id}.mp4")
        
        question_prompt = f"Based on the given video, reason and answer the single-choice question. Provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. The question is: {question}. The options are: {options}. Your answer:"


        # Predict anwer
        try:
            if "InternVL" in model_name:
                predicted_answer = predict_internvl(model, tokenizer, generation_config, question_prompt, video)
            elif "Qwen2.5-VL" in model_name or "VideoChat-R1" in model_name or "Video-R1" in model_name:
                predicted_answer = predict_Qwen_Family(llm, processor, sampling_params, question_prompt, video)
            elif "gemini" in model_name:
                predicted_answer = predict_gemini(api_key, model_name, question_prompt, video)
            # else:
            #    predicted_answer = predict_your_model(question_prompt, video)
        except:
           predicted_answer = "Wrong"
        print(predicted_answer)


        think_pattern = r'<think>\s*(.*?)\s*</think>'
        try:
            matches = re.findall(think_pattern, predicted_answer, re.DOTALL)
        except:
            matches = []
        if matches:
            thinking = matches[-1].strip()
        else:
            thinking = "WRONG"

        pattern = r'<answer>\s*(.*?)\s*</answer>'
        try:
            matches = re.findall(pattern, predicted_answer, re.DOTALL)
        except:
            matches = []
        if matches:
            choise = matches[-1].strip()
        else:
            choise = predicted_answer

        if 'A ' in choise or 'A:' in choise or '[A' in choise:
            predicted_answer = 'A'
        elif 'B ' in choise or 'B:' in choise or '[B' in choise:
            predicted_answer = 'B'
        elif 'C ' in choise or 'C:' in choise or '[C' in choise:
            predicted_answer = 'C'
        elif 'D ' in choise or 'D:' in choise or '[D' in choise:
            predicted_answer = 'D'
        elif 'E ' in choise or 'E:' in choise or '[E' in choise:
            predicted_answer = 'E'
        elif 'F ' in choise or 'F:' in choise or '[F' in choise:
            predicted_answer = 'F'
        elif 'A' in choise:
            predicted_answer = 'A'
        elif 'B' in choise:
            predicted_answer = 'B'
        elif 'C' in choise:
            predicted_answer = 'C'
        elif 'D' in choise:
            predicted_answer = 'D'
        elif 'E' in choise:
            predicted_answer = 'E'
        elif 'F' in choise:
            predicted_answer = 'F'
        else:
            predicted_answer = 'WRONG'

        if predicted_answer == correct_answer:
            correct_counts[q_type] += 1
        total_counts[q_type] += 1

        output_process.append({
            "video ID": video_id,
            "Question ID": quesion_id,
            "Question Type": q_type,
            "Question": item.get('Question'),
            "Options": item.get('Options'),
            "GT": correct_answer,
            "Explanation": explanation,  
            "Predicr Answer": predicted_answer,
            "Thinking": thinking,
            "Correct": predicted_answer == correct_answer,
        })
        
        with open(json_file_output, "w", encoding="utf-8") as f:
            json.dump(output_process, f, indent=2, ensure_ascii=False)


    accuracies = {q_type: (correct_counts[q_type] / total_counts[q_type] if total_counts[q_type] > 0 else 0) for q_type in question_types}
    total_correct = sum(correct_counts.values())
    total_questions = sum(total_counts.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    csv_file = os.path.join(output_path, f"Results-{model_name}.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = question_types + ['Overall Accuracy', 'Total Questions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        row = {q_type: f"{accuracies[q_type]:.2f}" for q_type in question_types}
        row['Overall Accuracy'] = f"{overall_accuracy:.2f}"
        row['Total Questions'] = total_questions 
        writer.writerow(row)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run model with specified name")
    parser.add_argument('--model_name', require=True, type=str)
    parser.add_argument('--model_path', default=None, type=str)
    args = parser.parse_args()
    
    args.model_name = "Intern-VL2.5-8B"
    model_name = args.model_name
    model_path = args.model_path

    video_path = 'Benchmark/videos/'
    benchmark = 'Benchmark/test_Video-Holmes.json'
    output_path = 'Results/'

    evaluate(video_path, benchmark, output_path, model_name, model_path)