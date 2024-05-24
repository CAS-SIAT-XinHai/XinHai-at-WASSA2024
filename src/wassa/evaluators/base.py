import json
import logging
import os
import re
from abc import abstractmethod
from collections import Counter

from datasets import load_dataset
from openai import OpenAI, OpenAIError
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


class BaseEvaluator(object):

    def __init__(self,
                 task, task_dir,
                 model_name, model_api_key, model_api_base,
                 evaluator_name, evaluator_api_key, evaluator_api_base):
        self.task = task
        self.task_dir = task_dir

        self.model_name = model_name
        self.model_client = OpenAI(
            api_key=model_api_key,
            base_url=model_api_base,
        )

        self.evaluator_name = evaluator_name
        self.evaluator_client = OpenAI(
            api_key=evaluator_api_key,
            base_url=evaluator_api_base,
        )

        # self.prompts_dir = prompts_dir

        # self.score_prompt = open(f"{self.prompts_dir}/score_prompt.txt").read()

        self.all_evaluate_results = {}
        self.evaluate_results = Counter()

    @property
    @abstractmethod
    def categories(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def task_mapping(self):
        raise NotImplementedError

    @staticmethod
    def chat_completion(client, model, messages):
        try:
            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.debug(f"Sending messages to {model}: {messages}")
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = chat_response.choices[0].message.content
            if content.strip():
                logger.debug(f"Get response from {model}: {content}")
                return content.strip()
            else:
                usage = chat_response.usage
                logger.error(f"Error response from {model}: {usage}")
        except OpenAIError as e:
            # Handle all OpenAI API errors
            logger.warning("*****************************************")
            logger.warning(f"Error response from {model}: {e}")

    def prompt_for_response(self, messages, num_retries=5):

        while num_retries:
            chat_response = self.chat_completion(self.evaluator_client, model=self.evaluator_name, messages=messages)
            if chat_response:
                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                if evaluate_ans:
                    evaluate_ans = evaluate_ans[0]
                    try:
                        d = json.loads(evaluate_ans)
                        if isinstance(d, dict) and len(d) > 0:
                            return d
                        else:
                            logger.error(f"Evaluation {evaluate_ans} error:", d)
                    except Exception as e:
                        logger.error(f"Evaluation {evaluate_ans} error:", e)
            # num_retries -= 1

    def run(self, split, n_shot, num_retries=5):
        pbar = tqdm(self.categories.keys(), desc="Processing subjects", position=0)
        results = {}
        logger.debug("=============================================================")
        outputs = {}
        for subject in pbar:
            dataset_name = self.task_mapping[self.categories[subject]['category']]
            dataset = load_dataset(
                path=os.path.join(self.task_dir, self.task),
                name=dataset_name,
            )
            pbar.set_postfix_str(self.categories[subject]["name"])
            eval_template = self.categories[subject]['template']

            category = self.categories[subject]['category']
            outputs.setdefault(category, {})

            label_key = self.categories[subject]['label_key']
            inputs, labels = [], []
            for i in trange(len(dataset[split]), desc=subject + "---" + dataset_name, position=1, leave=False):
                logger.debug("---------------------------------------------------------------")
                support_set = dataset["train"].shuffle().select(
                    range(min(n_shot, len(dataset["train"]))))
                target_data = dataset[split][i]
                subject_name = self.categories[subject]["name"]
                messages = eval_template.format_example(
                    target_data=target_data,
                    support_set=support_set,
                    subject_name=subject_name,
                    label_key=label_key,
                    dataset_name=dataset_name,
                    use_history=True,
                )

                response = self.prompt_for_response(messages, num_retries)

                outputs[category].setdefault(label_key, [])
                outputs[category][label_key].append(response[label_key])
