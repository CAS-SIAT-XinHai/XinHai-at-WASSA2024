import json
import logging
import re
from abc import abstractmethod
from collections import Counter
from typing import Dict, Tuple

from openai import OpenAI, OpenAIError

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

    @classmethod
    @abstractmethod
    def parse_example(
            cls,
            template,
            example: Dict[str, str],
            label_key,
            dataset_name
    ) -> Tuple[str, str]:
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

    @abstractmethod
    def run(self, split, n_shot, output_dir, num_retries=5):
        raise NotImplementedError
