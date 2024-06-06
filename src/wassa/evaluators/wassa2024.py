import csv
import logging
import os
import os.path
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset
from math import sqrt
from tqdm import tqdm, trange

from . import register_evaluator
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

# Participants are given a new dataset of empathic reactions to news stories and associated conversations which contains essays in reaction to news articles where there is harm to a person, group, or other (from Omitaomu and Tafreshi et al. 2023, similar to Buechel et al. 2018). Those essays contain Batson empathic concern and personal distress scores, as well as the Big Five personality (OCEAN) and Inter-Personal Index (IRI) scores of each user. This new dataset also contains conversations between two users that read the same article. Each of their speech turn has been annotated in perceived empathy, emotion polarity, and emotion intensity. The essays are between 300 and 800 characters in length. The conversations contains 11,788 speech turns. The dataset also includes the news articles and person-level demographic information (age, gender, ethnicity, income, education level).
#
# You can participate in five different tracks:
#
# Track 1: Empathy Prediction in Conversations (CONV-dialog), which consists in predicting the perceived empathy at the dialog-level
# Track 2: Empathy and Emotion Prediction in Conversations Turns (CONV-turn), which consists in predicting the perceived empathy, emotion polarity, and emotion intensity at the speech-turn-level in a conversation
# Track 3: Empathy Prediction (EMP), which consists in predicting both the empathy concern and the personal distress at the essay-level
# Track 4: Personality Prediction (PER), which consists in predicting the personality (openness, conscientiousness, extraversion, agreeableness, and emotional stability; OCEAN) of the essay writer, knowing all his/her essays, dialogs, and the news article from which they reacted


emotions = ['Hope/Sadness', 'Anger', 'Sadness', 'Neutral', 'Disgust/Sadness',
            'Anger/Disgust', 'Fear/Sadness', 'Joy', 'Hope', 'Joy/Neutral',
            'Disgust', 'Neutral/Sadness', 'Neutral/Surprise', 'Anger/Neutral',
            'Hope/Neutral', 'Surprise', 'Anger/Sadness', 'Fear', 'Anger/Joy',
            'Disgust/Fear', 'Fear/Neutral', 'Fear/Hope', 'Joy/Sadness',
            'Anger/Disgust/Sadness', 'Anger/Surprise', 'Disgust/Neutral',
            'Anger/Fear', 'Sadness/Surprise', 'Disgust/Surprise', 'Anger/Hope']

task_mapping = {
    'CONV': 'conversation',
    'EMP': 'essay',
    'EMO': 'essay',
    'PER': 'essay',
    'IRI': 'essay',
}
to_round = 4

nb_labels_CONVD = 1
nb_labels_CONVT = 3
nb_labels_EMP = 2
nb_labels_PER = 5


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def readFileToList(filename):
    # eprint("Reading data from",filename)
    lines = filename.readlines()
    result = []
    for x in lines:
        result.append(x.rstrip().split('\t'))
    filename.close()
    return result


def readCSVToList(filename):
    # eprint("Reading data from",filename)
    with open(filename.name, newline='') as f:
        reader = csv.reader(f)
        result = [list(row) for row in reader]
    return result


def readTSVToList(filename):
    # eprint("Reading data from",filename)
    with open(filename.name, newline='') as f:
        reader = csv.reader(f, delimiter="\t")
        result = [list(row) for row in reader]
    return result


def pearsonr(x, y):
    """
    Calculates a Pearson correlation coefficient.
    """

    assert len(x) == len(y), 'Prediction and gold standard does not have the same length...'

    xm = sum(x) / len(x)
    ym = sum(y) / len(y)

    xn = [k - xm for k in x]
    yn = [k - ym for k in y]

    r = 0
    r_den_x = 0
    r_den_y = 0
    for xn_val, yn_val in zip(xn, yn):
        r += xn_val * yn_val
        r_den_x += xn_val * xn_val
        r_den_y += yn_val * yn_val

    r_den = sqrt(r_den_x * r_den_y)

    if r_den:
        r = r / r_den
    else:
        r = 0

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)

    return round(r, to_round)


def calculate_pearson(gold, prediction):
    """
    gold/prediction are a list of lists [ emp pred , distress pred ]
    """

    # converting to float
    gold_float = []
    for k in gold:
        try:
            gold_float.append(float(k))
        except Exception as e:
            print(e)
            gold_float.append(0)

    prediction_float = []
    for k in prediction:
        try:
            prediction_float.append(float(k))
        except Exception as e:
            print(e)
            prediction_float.append(0)

    return pearsonr(gold_float, prediction_float)


def calculate_metrics(golds, predictions, task1, task2, task3, task4):
    """
    gold/prediction list of list of values : [ emp pred , distress pred , emo pred ]
    """

    start_label = 0
    if task1:
        start_label = 0
        gold_empathy = [k[start_label] for k in golds]
        prediction_empathy = [k[start_label] for k in predictions]
        pearson_CONVD = calculate_pearson(gold_empathy, prediction_empathy)
    else:
        pearson_CONVD = 0

    start_label = nb_labels_CONVD
    if task2:
        gold_convt, prediction_convt, pearson_convt = [], [], []
        for i in range(start_label, start_label + nb_labels_CONVT):
            gold_convt.append([k[i] for k in golds])
            prediction_convt.append([k[i] for k in predictions])
            pearson_convt.append(calculate_pearson(gold_convt[-1], prediction_convt[-1]))

        avg_pearson_CONVT = sum(pearson_convt) / len(pearson_convt)
        pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy = pearson_convt
    else:
        avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy = 0, 0, 0, 0

    start_label += nb_labels_CONVT
    if task3:
        gold_emp, prediction_emp, pearson_emp = [], [], []
        for i in range(start_label, start_label + nb_labels_EMP):
            gold_emp.append([k[i] for k in golds])
            prediction_emp.append([k[i] for k in predictions])
            pearson_emp.append(calculate_pearson(gold_emp[-1], prediction_emp[-1]))

        avg_pearson_EMP = sum(pearson_emp) / len(pearson_emp)
        pearson_empathy, pearson_distress = pearson_emp
    else:
        avg_pearson_EMP, pearson_empathy, pearson_distress = 0, 0, 0

    start_label += nb_labels_EMP
    if task4:
        gold_per, prediction_per, pearson_per = [], [], []
        for i in range(start_label, start_label + nb_labels_PER):
            gold_per.append([k[i] for k in golds])
            prediction_per.append([k[i] for k in predictions])
            pearson_per.append(calculate_pearson(gold_per[-1], prediction_per[-1]))

        avg_pearson_PER = sum(pearson_per) / len(pearson_per)
        person_ope, person_con, person_ext, person_agr, person_sta = pearson_per
    else:
        avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta = 0, 0, 0, 0, 0, 0

    return pearson_CONVD, avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy, avg_pearson_EMP, pearson_empathy, pearson_distress, avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta


def read_file(submission_path, nb_labels=2, nb_samp=10):
    """
    Read the tsv file
    """
    # unzipped submission data is always in the 'res' subdirectory
    if not os.path.exists(submission_path):
        print('Could not find submission file {0}'.format(submission_path))
        predictedList = [[0] * nb_labels] * nb_samp
        task = False
    else:
        submission_file = open(os.path.join(submission_path))
        # The 2 first columns
        predictedList = [k[:nb_labels] for k in readTSVToList(submission_file)]
        task = True

    return task, predictedList


def score(input_dir, output_dir):
    # unzipped reference data is always in the 'ref' subdirectory
    # read dev gold standard labels
    truth_file_CONVD = open(os.path.join(input_dir, 'ref', 'goldstandard_CONVD.tsv'))
    goldList_CONVD = [l[:nb_labels_CONVD] for l in readTSVToList(truth_file_CONVD)]
    nb_samp_CONVD = len(goldList_CONVD)

    truth_file_CONVT = open(os.path.join(input_dir, 'ref', 'goldstandard_CONVT.tsv'))
    goldList_CONVT = [l[:nb_labels_CONVT] for l in readTSVToList(truth_file_CONVT)]
    nb_samp_CONVT = len(goldList_CONVT)

    truth_file_EMP = open(os.path.join(input_dir, 'ref', 'goldstandard_EMP.tsv'))
    goldList_EMP = [l[:nb_labels_EMP] for l in readTSVToList(truth_file_EMP)]
    nb_samp_EMP = len(goldList_EMP)

    truth_file_PER = open(os.path.join(input_dir, 'ref', 'goldstandard_PER.tsv'))
    goldList_PER = [l[:nb_labels_PER] for l in readTSVToList(truth_file_PER)]
    nb_samp_PER = len(goldList_PER)

    goldList = [i + j + k + l for i, j, k, l in zip(goldList_CONVD, goldList_CONVT, goldList_EMP, goldList_PER)]

    # read predicyed labels
    submission_path = os.path.join(input_dir, 'res', 'predictions_CONVD.tsv')
    task1, predictedList_CONVD = read_file(submission_path=submission_path, nb_labels=nb_labels_CONVD,
                                           nb_samp=nb_samp_CONVD)

    submission_path = os.path.join(input_dir, 'res', 'predictions_CONVT.tsv')
    task2, predictedList_CONVT = read_file(submission_path=submission_path, nb_labels=nb_labels_CONVT,
                                           nb_samp=nb_samp_CONVT)

    submission_path = os.path.join(input_dir, 'res', 'predictions_EMP.tsv')
    task3, predictedList_EMP = read_file(submission_path=submission_path, nb_labels=nb_labels_EMP, nb_samp=nb_samp_EMP)

    submission_path = os.path.join(input_dir, 'res', 'predictions_PER.tsv')
    task4, predictedList_PER = read_file(submission_path=submission_path, nb_labels=nb_labels_PER, nb_samp=nb_samp_PER)

    predictedList = [i + j + k + l for i, j, k, l in
                     zip(predictedList_CONVD, predictedList_CONVT, predictedList_EMP, predictedList_PER)]

    if (len(goldList) != len(predictedList)):
        eprint("Number of labels is not aligned!")
        sys.exit(1)

    pearson_CONVD, avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy, avg_pearson_EMP, pearson_empathy, pearson_distress, avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta = calculate_metrics(
        goldList, predictedList, task1, task2, task3, task4)

    print("Printing results to:", output_dir + '/scores.txt')
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        str_to_write = ''
        # Not sure of that. Useful if the participant want to do only one subtask. Need to see if the leaderboard of the subtask does not update if there are nothing on score.txt
        if task1:
            str_to_write += "Track 1 (CONVD): Pearson Correlation (perceived empathy): {0}\n".format(pearson_CONVD)
        if task2:
            str_to_write += "Track 2 (CONVT): Averaged Pearson Correlations: {0}\n\tEmotion: {1}\n\tEmotion Polarity: {2}\n\tEmpathy: {3}\n".format(
                avg_pearson_CONVT, pearson_t_emotion, pearson_t_emotionpolarity, pearson_t_empathy)
        if task3:
            str_to_write += "Track 3 (EMP): Averaged Pearson Correlations: {0}\n\tEmpathy: {1}\n\tDistress: {2}\n".format(
                avg_pearson_EMP, pearson_empathy, pearson_distress)
        if task4:
            str_to_write += "Track 4 (PER): Averaged Pearson Correlations: {0}\n\tOpenness: {1}\n\tConscientiousness: {2}\n\tExtraversion: {3}\n\tAgreeableness: {4}\n\tStability: {5}\n".format(
                avg_pearson_PER, person_ope, person_con, person_ext, person_agr, person_sta)
        output_file.write(str_to_write)


@dataclass
class WASSA2024EvalTemplate:
    name: str
    system: str
    # article: str
    # essay: str
    instruction: str

    def parse_example(
            self,
            example: Dict[str, str],
            label_key,
            dataset_name
    ) -> Tuple[str, str]:
        # print(json.dumps(example, indent=2))
        article = "\nArticle: {article}".format(article=example['article'])
        label = example[label_key]
        if dataset_name == 'conversation':
            conversation = "\nConversation between speakers: {history}".format(history=example['history'])
            response = "\nResponse by Speaker {speaker_id}: {text}".format(speaker_id=example['speaker_id'],
                                                                           text=example['text'])
            if isinstance(label, float):
                label = "{:.2f}".format(label)
            return "".join([article] + [conversation] + [response] + [self.instruction]), label
        else:
            essay = "\nEssay written by Speaker {speaker_id}: {essay}".format(speaker_id=example['speaker_id'],
                                                                              essay=example['essay'])
            if isinstance(label, float):
                label = "{:.2f}".format(label)
            return "".join([article] + [essay] + [self.instruction]), label

    def format_example(
            self,
            target_data: Dict[str, str],
            support_set: "Dataset",
            subject_name: str,
            label_key: str,
            dataset_name: str,
            use_history: bool,
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        query, resp = self.parse_example(target_data, label_key=label_key, dataset_name=dataset_name)
        history = [self.parse_example(support_set[k], label_key=label_key, dataset_name=dataset_name) for k in
                   range(len(support_set))]

        if len(history):
            temp = history.pop(0)
            history.insert(0, (self.system.format(subject=subject_name) + temp[0], temp[1]))
        else:
            query = self.system.format(subject=subject_name) + query

        if not use_history:
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        # print(history, query)
        return query.strip(), resp, history


@register_evaluator("wassa2024", "multi_scorer")
class WASSA2024MultiScorerEvaluator(BaseEvaluator):

    def __init__(self,
                 task_dir,
                 model_name,
                 model_api_key,
                 model_api_base,
                 evaluator_name,
                 evaluator_api_key,
                 evaluator_api_base):
        super().__init__("wassa2024", task_dir,
                         model_name, model_api_key, model_api_base, evaluator_name, evaluator_api_key,
                         evaluator_api_base)

    def task_mapping(self):
        return {
            'CONVD': 'CONVD',
            'CONVT': 'CONVT',
            'EMP': 'EMP',
            'PER': 'PER',
        }

    @property
    def categories(self):
        return {
            "CONVD": {
                "name": "at the speech-turn-level",
                "label_key": ["EmotionalPolarity", "Emotion", "Empathy"],
                "template": WASSA2024EvalTemplate(
                    name="conversation",
                    instruction="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict:\n"
                                "1. EmotionalPolarity: as a score from the range : 0 to 2 ;\n"
                                "2. Emotion: as a score from the range : 1 to 5 ;\n"
                                "3. Empathy: as a score from the range : 1 to 5 .\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {\"EmotionalPolarity\": 1.3, \"Emotion\": 3.6, \"Empathy\": 4.3}\n"
                          "```",
                ),
                "category": "CONV"
            },
            "CONVT": {
                "name": "Emotion at the essay-level",
                "label_key": ["emotion"],
                "template": WASSA2024EvalTemplate(
                    name="essay_emotion",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict Emotion at the essay-level from one or more emotion labels from the Ekman’s six basic emotions (sadness, joy, disgust, surprise, anger, or fear) as well as neutral. The essay expresses the emotion:\n\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {\"emotion\": \"disgust\"}\n"
                          "```",
                ),
                "category": "EMO"
            },
            "EMP": {
                "name": "Empathy at the essay-level",
                "label_key": ["empathy", "distress"],
                "template": WASSA2024EvalTemplate(
                    name="essay",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Empathy score is an average of 7-point scale ratings, representing each of the following states (warm, tender, sympathetic,softhearted, moved, compassionate). Try to predict:\n"
                                "1. empathy as a score from the range : 1 to 7:\n"
                                "2. distress as a score from the range : 1 to 7:\n"
                                "\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {\"empathy\": 5.8, \"distress\": 2.5}\n"
                          "```",
                ),
                "category": "EMP"
            },
            "PER": {
                "name": "essay writer",
                "label_key": [
                    "personality_conscientiousness",
                    "personality_openness",
                    "personality_extraversion",
                    "personality_agreeableness",
                    "personality_stability"
                ],
                "template": WASSA2024EvalTemplate(
                    name="writer_conscientiousness",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict metrics of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n"
                                "1. personality_conscientiousness:\n"
                                "2. personality_openness\n"
                                "3. personality_extraversion\n"
                                "4. personality_agreeableness\n"
                                "5. personality_stability\n"
                                "\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {\"personality_conscientiousness\": 1.3, \"personality_openness\": 3.6, \"personality_extraversion\": 4.3, \"personality_agreeableness\": 3.6, \"personality_stability\": 4.3}\n"
                          "```",
                ),
                "category": "PER"
            },
            # "IRI": {
            #     "name": "Perspective-taking of the essay writer",
            #     "label_key": [
            #         "iri_perspective_taking",
            #         "iri_personal_distress",
            #         "iri_fantasy",
            #         "iri_empathetic_concern"
            #     ],
            #     "template": WASSA2023EvalTemplate(
            #         name="writer_perspective_taking",
            #         instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict metrics of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n"
            #                     "1. iri_perspective_taking\n"
            #                     "2. iri_personal_distress\n"
            #                     "3. iri_fantasy\n"
            #                     "4. iri_empathetic_concern\n"
            #                     "\n",
            #         input="Provide your evaluation in JSON format, as shown in the example below.\n"
            #               "Example of Evaluation Output:\n"
            #               "```json\n"
            #               "  {\"iri_perspective_taking\": 1.3, \"iri_personal_distress\": 3.6, \"iri_fantasy\": 4.3, \"iri_empathetic_concern\": 3.6}\n"
            #               "```",
            #     ),
            #     "category": "IRI"
            # }
        }

    @classmethod
    def parse_example(
            cls,
            template,
            example: Dict[str, str],
            label_key,
            dataset_name
    ) -> Tuple[str, str]:
        article = "[Article]\n{article}\n[End of Article]".format(article=example['article'])
        label = {lk: "{:.2f}".format(example[lk]) if isinstance(example[lk], float) else example[lk] for lk in
                 label_key}
        if dataset_name == 'conversation':
            conversation = "[Conversation]\n{history}\n[End of Conversation]".format(history=example['history'])
            response = "[Response by Speaker {speaker_id}]\n{text}\n[End of Response by Speaker {speaker_id}]".format(
                speaker_id=example['speaker_id'],
                text=example['text'])
            query, resp = "\n\n".join(
                [article] + [conversation] + [response] + [template.instruction] + [template.input]), json.dumps(label)
        else:
            essay = "[Essay by Speaker {speaker_id}]\n{essay}\n[End of Essay by Speaker {speaker_id}]".format(
                speaker_id=example['speaker_id'],
                essay=example['essay'])
            query, resp = "\n\n".join([article] + [essay] + [template.instruction] + [template.input]), json.dumps(
                label)
        logger.debug(query)
        logger.debug(resp)
        return query, resp

    def run(self, split, n_shot, output_dir, num_retries=5):
        ref_dir = os.path.join(output_dir, split, "ref")
        res_dir = os.path.join(output_dir, split, "res")
        ret_dir = os.path.join(output_dir, split, "ret")
        [os.makedirs(d, exist_ok=True) for d in [ref_dir, res_dir, ret_dir]]

        pbar = tqdm(self.categories.keys(), desc="Processing subjects", position=0)
        logger.debug("=============================================================")
        for subject in pbar:
            dataset_name = self.task_mapping[self.categories[subject]['category']]
            dataset = load_dataset(
                path=os.path.join(self.task_dir, self.task),
                name=dataset_name,
            )
            pbar.set_postfix_str(self.categories[subject]["name"])
            eval_template = self.categories[subject]['template']

            category = self.categories[subject]['category']
            label_key = self.categories[subject]['label_key']

            task_file = os.path.join(res_dir, f'predictions_{category}.tsv')
            if not os.path.exists(task_file):
                with open(task_file, "w") as fd:
                    for i in trange(len(dataset[split]), desc=subject + "---" + dataset_name, position=1, leave=False):
                        logger.debug("---------------------------------------------------------------")
                        support_set = dataset["train"].shuffle().select(
                            range(min(n_shot, len(dataset["train"]))))
                        target_data = dataset[split][i]
                        logger.debug(f"Example: {target_data}")
                        subject_name = self.categories[subject]["name"]
                        messages = eval_template.format_example(
                            target_data=target_data,
                            support_set=support_set,
                            subject_name=subject_name,
                            label_key=label_key,
                            dataset_name=dataset_name,
                            use_history=True,
                            parse_func=self.parse_example
                        )

                        result = None
                        while not result:
                            response = self.prompt_for_response(messages, num_retries)
                            try:
                                result = [response[k] for k in label_key]
                                logger.debug(result)
                            except Exception as e:
                                logger.warning(f"Error response for {subject}: {response}, {e}")

                        fd.write(
                            "\t".join(map(lambda x: x if isinstance(x, str) else "{:.2f}".format(x), result)) + "\n")

        if split == "validation":
            with zipfile.ZipFile(os.path.join(self.task_dir, self.task, f'{self.task}.zip')) as zd:
                for filename in zd.namelist():
                    if filename in ['goldstandard_dev.tsv', 'goldstandard_CONV_dev.tsv']:
                        with open(os.path.join(ref_dir, filename.replace("_dev", "")), 'wb') as fd:
                            with zd.open(filename) as f:
                                fd.write(f.read())
            score(os.path.join(output_dir, split), ret_dir)
