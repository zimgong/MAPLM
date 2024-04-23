from PIL import Image
import os
import json
import torch
import random
from tqdm import tqdm
from typing import List
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms

from tools import utils

class Arguments:
    def __init__(self):
        self.data_root = 'data/maplm_v0.1'
        self.output_dir = 'runs'
        self.test_split = 'test'
        self.test_number = -1
        self.exp_label = 'clip'
        self.random_seed = 42
        self.debug = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arguments = Arguments()
    random.seed(arguments.random_seed)
    frames, frame_ids = utils.load_data(arguments)
    result_file_name = utils.get_result_file(arguments)
    results = dict(
        question_overall=utils.acc_counter(),
        frame_overall=utils.acc_counter(),
    )
    question_prompt_map = {
        "What kind of road scene is it in the images?": "A photo of a ",
        "What is the point cloud data quality in current road area of this image?": "The point cloud is ",
        "How many lanes in current road?": ["There are ", " lanes in this road."],
        "Is there any road cross, intersection or lane change zone in the main road?": ""
    }

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    transform = transforms.Compose([transforms.ToTensor()])

    for i, frame_id in enumerate(tqdm(frame_ids)):
        frame = frames[frame_id]

        image_list = frame['image']
        qas = frame['qa']

        corrects = []

        image_path = f'{arguments.data_root}/{arguments.test_split}/{frame_id}/{image_list[1]}'
        image = Image.open(image_path)
        image = transform(image).to(device)
        for j, qa in enumerate(qas):
            if qa['task'] != 'closed choice':
                continue
            question = qa['question']
            choices: List[str] = qa['choices']
            true_answer: int = qa['answer']

            text = []
            if question == "What kind of road scene is it in the images?":
                for choice in choices:
                    if "None" in choice:
                        text.append(choice)
                    elif "Round" in choice:
                        text.append("None")
                    else:
                        text.append(question_prompt_map[question] + choice)
            elif question == "What is the point cloud data quality in current road area of this image?":
                for choice in choices:
                    text.append(question_prompt_map[question] + choice.replace('V', 'v').replace('N', 'n'))
            elif question == "How many lanes in current road?":
                for choice in choices:
                    text.append(question_prompt_map[question][0] + choice + question_prompt_map[question][1])
            elif question == "Is there any road cross, intersection or lane change zone in the main road?":
                for choice in choices:
                    if "Yes" in choice:
                        text.append(choice.replace('Yes, t', 'T'))
                    else:
                        text.append(choice)
            else:
                continue
                
            inputs = processor(text, images=image, return_tensors="pt", padding=True)
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            inputs['pixel_values'] = inputs['pixel_values'].to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            guess = torch.argmax(probs, dim=1).item()

            if question not in results:
                results[question] = utils.acc_counter()

            correct = bool(guess == true_answer)
            corrects.append(correct)

            results[question]['total'] += 1
            results[question]['correct'] += int(correct)
            results['question_overall']['total'] += 1
            results['question_overall']['correct'] += int(correct)

        results['frame_overall']['total'] += 1
        results['frame_overall']['correct'] += int(all(corrects))

    acc_dict = utils.compute_acc(results)
    print(json.dumps(acc_dict, indent=4, sort_keys=True))
    print(json.dumps(results, indent=4, sort_keys=True))
    if not os.path.exists(arguments.output_dir):
        os.makedirs(arguments.output_dir)
    with open(arguments.output_dir + '/' + result_file_name, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()