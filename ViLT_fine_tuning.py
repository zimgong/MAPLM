from PIL import Image
import os
import torch
import random
import wandb
from tqdm import tqdm
from typing import List
from transformers import ViltForQuestionAnswering, ViltProcessor
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from tools import utils


class Arguments:
    def __init__(self):
        self.data_root = 'data/maplm_v0.1'
        self.output_dir = 'runs'
        self.test_split = 'train'
        self.test_number = 2000
        self.exp_label = 'transformer'
        self.random_seed = 42
        self.debug = False
        self.batch_size = 4
        self.lr = 5e-5
        self.num_epochs = 5
        self.report_to = 'wandb'


labels = [
    "Normal city road.",
    "Construction road.",
    "Undeveloped road.",
    "Road mark repainting.",
    "Roundabout.",
    "None of the above."
]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}


class MAPLMDataset(torch.utils.data.Dataset):
    """MAPLM dataset."""

    def __init__(self, frames, frame_ids, processor, arguments):
        self.frames = frames
        self.frame_ids = frame_ids
        self.processor = processor
        self.arguments = arguments

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        # get image + text
        frame_id = self.frame_ids[idx]
        frame = self.frames[frame_id]
        questions = frame['qa']
        image_list = frame['image']
        image_path = f'{self.arguments.data_root}/{self.arguments.test_split}/{frame_id}/{image_list[1]}'
        image = Image.open(image_path)
        text = questions[0]['question']

        encoding = self.processor(
            image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        # add labels
        choices = questions[0]['choices']
        labels = [label2id[choice] for choice in choices]
        scores = [1 if label == questions[0]
                  ['answer'] else 0.1 for label in labels]
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(id2label))
        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arguments = Arguments()

    # training_args = TrainingArguments(
    #     output_dir='models',
    #     overwrite_output_dir=True,
    #     learning_rate=arguments.lr,
    #     per_device_train_batch_size=arguments.batch_size,
    #     per_device_eval_batch_size=arguments.batch_size,
    #     do_eval=True,
    #     seed=arguments.random_seed,
    #     evaluation_strategy='steps',
    #     eval_steps=50,
    #     save_strategy='steps',
    #     save_steps=100,
    #     num_train_epochs=100,
    #     logging_dir='logs',
    #     load_best_model_at_end=True,
    #     metric_for_best_model='eval_f1',
    #     greater_is_better=True,
    #     report_to=arguments.report_to,
    #     logging_steps=50,
    #     fp16=False,
    #     gradient_accumulation_steps=1,
    #     lr_scheduler_type='linear'
    # )

    torch.manual_seed(arguments.random_seed)
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

    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                     id2label=id2label,
                                                     label2id=label2id)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model.to(device)

    dataset_train = MAPLMDataset(frames=frames,
                           frame_ids=frame_ids,
                           processor=processor,
                            arguments=arguments)
    
    test_split = arguments.test_split
    arguments.test_split = 'val'
    frames_dev, frame_ids_dev = utils.load_data(arguments)
    dataset_dev = MAPLMDataset(frames=frames_dev,
                            frame_ids=frame_ids_dev,
                            processor=processor,
                            arguments=arguments)
    arguments.test_split = test_split

    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        pixel_values = [item['pixel_values'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        # create padded pixel values and corresponding pixel mask
        encoding = processor.image_processor.pad(
            pixel_values, return_tensors="pt")

        # create new batch
        batch = {}
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['token_type_ids'] = torch.stack(token_type_ids)
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = torch.stack(labels)

        return batch

    train_dataloader = DataLoader(
        dataset_train, collate_fn=collate_fn, batch_size=arguments.batch_size, shuffle=True)

    dev_dataloader = DataLoader(
        dataset_dev, collate_fn=collate_fn, batch_size=arguments.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr)

    wandb.init(
        project="maplm",
        config={
            "learning_rate": arguments.lr,
            "architecture": "ViLT",
            "dataset": "maplm",
            "num_epochs": arguments.num_epochs,
        }
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     eval_dataset=dataset,
    #     data_collator=None,
    #     compute_metrics=None
    # )

    for epoch in range(arguments.num_epochs):
        model.train()
        arguments.test_split = 'train'
        print(f"Epoch {epoch}")

        loss_sum = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

            if step % 100 == 0 and step > 0:
                wandb.log({"loss": loss_sum / 100})
                loss_sum = 0

        model.eval()
        arguments.test_split = 'val'
        correct = 0
        total = 0
        loss_sum = 0
        for step, batch in enumerate(tqdm(dev_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch['labels']
            ground_truth = torch.argmax(labels, dim=1)
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == ground_truth).item()
            total += len(ground_truth)
            loss = outputs.loss
            loss_sum += loss.item()
        wandb.log({"accuracy": correct / total})
        wandb.log({"loss": loss_sum / len(dev_dataloader)})

    model.eval()
    wandb.finish()


if __name__ == "__main__":
    main()
