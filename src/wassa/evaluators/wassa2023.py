import json
import logging
import os
import os.path
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple

from datasets import load_dataset
from math import sqrt
from tqdm import tqdm, trange

from . import register_evaluator
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

# Participants are given a new dataset of empathic reactions to news stories and associated conversations which contains essays in reaction to news articles where there is harm to a person, group, or other (from Omitaomu et al. 2022, similar to Buechel et al. 2018).
# Those essays contain Batson empathic concern and personal distress scores, as well as the Big Five personality and Inter-Personal Index (IRI) scores of each user.
# This new dataset also contains conversations between two users that read the same article.
# Each of their speech turn has been annotated in perceived empathy, emotion polarity and emotion intensity.
# The essays are between 300 and 800 characters in length.
# The conversations contain in average 23 speech turns.
# The dataset also includes the news articles and person-level demographic information (age, gender, ethnicity, income, education level).
#
# Track 1: Empathy and Emotion Prediction in Conversations (CONV), which consists in predicting the perceived empathy, emotion polarity and emotion intensity at the speech-turn-level in a conversation
# Track 2: Empathy Prediction (EMP), which consists in predicting both the empathy concern and the personal distress at the essay-level
# Track 3: Emotion Classification (EMO), which consists in predicting the emotion at the essay-level
# Track 4: Personality Prediction (PER), which consists in predicting the personality of the essay writer, knowing all his/her essays and the news article from which they reacted
# Track 5: Interpersonal Reactivity Index Prediction (IRI), which consists in predicting the personality of the essay writer, knowing all his/her essays and the news article from which they reacted

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


def calculatePRF(gold, prediction):
    """
    gold/prediction list of list of emo predictions
    """
    # initialize counters
    labels = set(gold + prediction)
    tp = dict.fromkeys(labels, 0.0)
    fp = dict.fromkeys(labels, 0.0)
    fn = dict.fromkeys(labels, 0.0)
    precision = dict.fromkeys(labels, 0.0)
    recall = dict.fromkeys(labels, 0.0)
    f = dict.fromkeys(labels, 0.0)
    # check every element
    for g, p in zip(gold, prediction):
        # TP
        if (g == p):
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    # print("Label\tTP\tFP\tFN\tP\tR\tF")
    for label in labels:
        recall[label] = 0.0 if (tp[label] + fn[label]) == 0.0 else (tp[label]) / (tp[label] + fn[label])
        precision[label] = 1.0 if (tp[label] + fp[label]) == 0.0 else (tp[label]) / (tp[label] + fp[label])
        f[label] = 0.0 if (precision[label] + recall[label]) == 0 else (2 * precision[label] * recall[label]) / (
                precision[label] + recall[label])
        microrecall = (sum(tp.values())) / (sum(tp.values()) + sum(fn.values()))
        microprecision = (sum(tp.values())) / (sum(tp.values()) + sum(fp.values()))
        microf = 0.0 if (microprecision + microrecall) == 0 else (2 * microprecision * microrecall) / (
                microprecision + microrecall)
    # Macro average
    macrorecall = sum(recall.values()) / len(recall)
    macroprecision = sum(precision.values()) / len(precision)
    macroF = sum(f.values()) / len(f)

    accuracy = 0
    for label in labels:
        accuracy += tp[label]

    accuracy = accuracy / len(gold)

    return round(microrecall, to_round), round(microprecision, to_round), round(microf, to_round), round(macrorecall,
                                                                                                         to_round), round(
        macroprecision, to_round), round(macroF, to_round), round(accuracy, to_round)


def calculatePRF_MLabel(gold, prediction):
    """
    gold/prediction list of list of emo predictions
    """
    # initialize counters
    # labels = set(gold+prediction)

    gold = [k.lower().split('/') for k in gold]
    prediction = [k.lower().split('/') for k in prediction]

    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import precision_recall_fscore_support, jaccard_score
    mlb = MultiLabelBinarizer()
    mlb.fit(gold)

    y_true = mlb.transform(gold)
    y_pred = mlb.transform(prediction)

    microprecision, microrecall, microf, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    macroprecision, macrorecall, macroF, s = precision_recall_fscore_support(y_true, y_pred, average='macro')

    accuracy = jaccard_score(y_true, y_pred, average='micro')

    return round(microrecall, to_round), round(microprecision, to_round), round(microf, to_round), round(macrorecall,
                                                                                                         to_round), round(
        macroprecision, to_round), round(macroF, to_round), round(accuracy, to_round)


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
    gold = [float(k) for k in gold]
    prediction = [float(k) for k in prediction]

    return pearsonr(gold, prediction)


def calculate_metrics(golds, predictions, task1, task2, task3, task4):
    """
    gold/prediction list of list of values : [ emp pred , distress pred , emo pred ]
    """
    if task1:
        gold_empathy = [k[0] for k in golds]
        prediction_empathy = [k[0] for k in predictions]
        pearson_empathy = calculate_pearson(gold_empathy, prediction_empathy)

        gold_distress = [k[1] for k in golds]
        prediction_distress = [k[1] for k in predictions]
        pearson_distress = calculate_pearson(gold_distress, prediction_distress)
        avg_pearson = (pearson_empathy + pearson_distress) / 2
    else:
        avg_pearson, pearson_empathy, pearson_distress = 0, 0, 0

    if task2:
        gold_emo = [k[2] for k in golds]
        prediction_emo = [k[2] for k in predictions]

        microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy = calculatePRF_MLabel(
            gold_emo, prediction_emo)
    else:
        microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy = 0, 0, 0, 0, 0, 0, 0

    if task3:
        gold_per = []
        prediction_per = []
        pearson_per = []
        for i in range(3, 8):
            gold_per.append([k[i] for k in golds])
            prediction_per.append([k[i] for k in predictions])
            pearson_per.append(calculate_pearson(gold_per[-1], prediction_per[-1]))

        avg_pearson_PER = sum(pearson_per) / len(pearson_per)
    else:
        avg_pearson_PER = 0

    if task4:
        gold_iri = []
        prediction_iri = []
        pearson_iri = []
        for i in range(8, len(golds[0])):
            gold_iri.append([k[i] for k in golds])
            prediction_iri.append([k[i] for k in predictions])
            pearson_iri.append(calculate_pearson(gold_iri[-1], prediction_iri[-1]))

        avg_pearson_IRI = sum(pearson_iri) / len(pearson_iri)
    else:
        avg_pearson_IRI = 0

    return avg_pearson, pearson_empathy, pearson_distress, microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy, avg_pearson_PER, avg_pearson_IRI


def calculate_metrics_CONV(golds, predictions, task5):
    """
    gold/prediction list of list of values : [ emp pred , distress pred , emo pred ]
    """

    if task5:
        gold_CONV = []
        prediction_CONV = []
        pearson_CONV = []
        for i in range(3):
            gold_CONV.append([k[i] for k in golds])
            prediction_CONV.append([k[i] for k in predictions])
            pearson_CONV.append(calculate_pearson(gold_CONV[-1], prediction_CONV[-1]))
    else:
        pearson_CONV = [0, 0, 0]

    avg_pearson_CONV = sum(pearson_CONV) / len(pearson_CONV)
    pearson_CONV_EMOP, pearson_CONV_EMOI, pearson_CONV_EMP = pearson_CONV

    return avg_pearson_CONV, pearson_CONV_EMP, pearson_CONV_EMOP, pearson_CONV_EMOI


def read_file(submission_path, nb_labels=2, nb_samp=10):
    """
    Read the tsv file
    """
    # unzipped submission data is always in the 'res' subdirectory
    if not os.path.exists(submission_path):
        print('Could not find submission file {0}'.format(submission_path))
        predictedList_EMP = [[0] * nb_labels] * nb_samp
        task1 = False
    else:
        submission_file = open(os.path.join(submission_path))
        # The 2 first columns
        predictedList_EMP = [k[:nb_labels] for k in readFileToList(submission_file)]
        task1 = True

    return task1, predictedList_EMP


nb_labels_EMP = 2
nb_labels_EMO = 1
nb_labels_PER = 5
nb_labels_IRI = 4
nb_labels_CONV = 3


def score(input_dir, output_dir):
    # unzipped reference data is always in the 'ref' subdirectory
    truth_file = open(os.path.join(input_dir, 'ref', 'goldstandard.tsv'))
    goldList = readFileToList(truth_file)
    nb_samp = len(goldList)

    truth_file_CONV = open(os.path.join(input_dir, 'ref', 'goldstandard_CONV.tsv'))
    goldList_CONV = readFileToList(truth_file_CONV)
    nb_samp_CONV = len(goldList_CONV)

    submission_path = os.path.join(input_dir, 'res', 'predictions_EMP.tsv')
    task1, predictedList_EMP = read_file(submission_path=submission_path, nb_labels=nb_labels_EMP, nb_samp=nb_samp)

    submission_path = os.path.join(input_dir, 'res', 'predictions_EMO.tsv')
    task2, predictedList_EMO = read_file(submission_path=submission_path, nb_labels=nb_labels_EMO, nb_samp=nb_samp)
    if goldList[0][2] == 'Nolabel': task2 = False

    submission_path = os.path.join(input_dir, 'res', 'predictions_PER.tsv')
    task3, predictedList_PER = read_file(submission_path=submission_path, nb_labels=nb_labels_PER, nb_samp=nb_samp)

    submission_path = os.path.join(input_dir, 'res', 'predictions_IRI.tsv')
    task4, predictedList_IRI = read_file(submission_path=submission_path, nb_labels=nb_labels_IRI, nb_samp=nb_samp)

    submission_path = os.path.join(input_dir, 'res', 'predictions_CONV.tsv')
    task5, predictedList_CONV = read_file(submission_path=submission_path, nb_labels=nb_labels_CONV,
                                          nb_samp=nb_samp_CONV)

    predictedList = [i + j + k + l for i, j, k, l in
                     zip(predictedList_EMP, predictedList_EMO, predictedList_PER, predictedList_IRI)]

    if (len(goldList) != len(predictedList)):
        eprint("Number of labels is not aligned!")
        sys.exit(1)

    if task5 and (len(goldList_CONV) != len(predictedList_CONV)):
        eprint("Number of labels for CONV is not aligned!")
        sys.exit(1)

    avg_pearson, pearson_empathy, pearson_distress, micror, microp, microf, macror, macrop, macrof, accuracy, avg_pearson_PER, avg_pearson_IRI = calculate_metrics(
        goldList, predictedList, task1, task2, task3, task4)

    avg_pearson_CONV, pearson_CONV_EMP, pearson_CONV_EMOP, pearson_CONV_EMOI = calculate_metrics_CONV(goldList_CONV,
                                                                                                      predictedList_CONV,
                                                                                                      task5)

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        str_to_write = ''
        # Not sure of that. Useful if the participant want to do only one subtask. Need to see if the leaderboard of the subtask does not update if there are nothing on score.txt
        if task1:
            str_to_write += "Averaged Pearson Correlations: {0}\nEmpathy Pearson Correlation: {1}\nDistress Pearson Correlation: {2}\n".format(
                avg_pearson, pearson_empathy, pearson_distress)
        if task2:
            str_to_write += "Macro F1-Score: {5}\nMicro Recall: {0}\nMicro Precision: {1}\nMicro F1-Score: {2}\nMacro Recall: {3}\nMacro Precision: {4}\nMicro Jaccard: {6}\n".format(
                micror, microp, microf, macror, macrop, macrof, accuracy)
        if task3:
            str_to_write += "PER Pearson Correlations: {0}\n".format(avg_pearson_PER)
        if task4:
            str_to_write += "IRI Pearson Correlations: {0}\n".format(avg_pearson_IRI)
        if task5:
            str_to_write += "Conversation Pearson Correlations: {0}\nConversation Empathy Pearson Correlation: {1}\nConversation Emotional Polarity Pearson Correlation: {2}\nConversation Emotional Intensity Pearson Correlation: {3}\n".format(
                avg_pearson_CONV, pearson_CONV_EMP, pearson_CONV_EMOP, pearson_CONV_EMOI)
        output_file.write(str_to_write)


@dataclass
class WASSA2023EvalTemplate:
    name: str
    instruction: str
    # article: str
    # essay: str
    input: str

    def format_example(
            self,
            target_data: Dict[str, str],
            support_set: "Dataset",
            subject_name: str,
            label_key: str,
            dataset_name: str,
            use_history: bool,
            parse_func
    ) -> list[list[dict[str, str] | dict[str, str]] | dict[str, str]]:
        """
        use_history: whether to use history or not.
        """
        query, resp = parse_func(self, target_data, label_key=label_key, dataset_name=dataset_name)
        history = [parse_func(self, support_set[k], label_key=label_key, dataset_name=dataset_name) for k in
                   range(len(support_set))]

        if len(history):
            temp = history.pop(0)
            history.insert(0, (self.instruction.format(subject=subject_name) + temp[0], temp[1]))
        else:
            query = self.instruction.format(subject=subject_name) + query

        messages = []

        if not use_history:
            query = "\n\n".join(["".join(item) for item in history] + [query])
        else:
            for u, a in history:
                messages.append([{
                    "role": "user",
                    "content": u
                }, {
                    "role": "assistant",
                    "content": a
                }])
        # print(history, query)
        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )
        return messages


@register_evaluator("wassa2023", "baseline")
class WASSA2023Evaluator(BaseEvaluator):

    def __init__(self,
                 task_dir,
                 model_name,
                 model_api_key,
                 model_api_base,
                 evaluator_name,
                 evaluator_api_key,
                 evaluator_api_base):
        super().__init__("wassa2023", task_dir,
                         model_name, model_api_key, model_api_base, evaluator_name, evaluator_api_key,
                         evaluator_api_base)

    @property
    def categories(self):
        return {
            "conversation_emotion_polarity": {
                "name": "Emotion Polarity at the speech-turn-level",
                "label_key": "EmotionalPolarity",
                "template": WASSA2023EvalTemplate(
                    name="conversation_emotion_polarity",
                    instruction="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} as a score from the range : 0 to 2:\n\n",
                    input="\n The score for the response on emotion polarity is ",
                ),
                "category": "CONV"
            },
            "conversation_emotion_intensity": {
                "name": "Emotion Intensity at the speech-turn-level",
                "label_key": "Emotion",
                "template": WASSA2023EvalTemplate(
                    name="conversation_emotion_intensity",
                    instruction="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} as a score from the range : 1 to 5:\n\n",
                    input="\n The score for the response on emotion intensity is ",
                ),
                "category": "CONV"
            },
            "conversation_empathy": {
                "name": "Empathy at the speech-turn-level",
                "label_key": "Empathy",
                "template": WASSA2023EvalTemplate(
                    name="conversation_empathy",
                    instruction="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} as a score from the range : 1 to 5:\n\n",
                    input="\n The score for the response on empathy is ",
                ),
                "category": "CONV"
            },
            "essay_empathy": {
                "name": "Empathy at the essay-level",
                "label_key": "empathy",
                "template": WASSA2023EvalTemplate(
                    name="essay_empathy",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Empathy score is an average of 7-point scale ratings, representing each of the following states (warm, tender, sympathetic,softhearted, moved, compassionate). Try to predict {subject} as a score from the range : 1 to 7:\n\n",
                    input="\n The score for the response on empathy is ",
                ),
                "category": "EMP"
            },
            "essay_distress": {
                "name": "Personal Distress at the essay-level",
                "label_key": "distress",
                "template": WASSA2023EvalTemplate(
                    name="essay_distress",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Distress score is an average of 7-point scale ratings, representing each of the following states (worried, upset, troubled, perturbed, grieved, disturbed, alarmed,distressed). Try to predict {subject} as a score from the range : 1 to 7:\n\n",
                    input="\n The score for the response on distress is ",
                ),
                "category": "EMP"
            },
            "essay_emotion": {
                "name": "Emotion at the essay-level",
                "label_key": "emotion",
                "template": WASSA2023EvalTemplate(
                    name="essay_emotion",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} from one or more emotion labels from the Ekman’s six basic emotions (sadness, joy, disgust, surprise, anger, or fear) as well as neutral:\n\n",
                    input="\nThe essay expresses the emotion ",
                ),
                "category": "EMO"
            },
            "writer_conscientiousness": {
                "name": "Conscientiousness of the essay writer",
                "label_key": "personality_conscientiousness",
                "template": WASSA2023EvalTemplate(
                    name="writer_conscientiousness",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    input="\nThe score for the Conscientiousness of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_openness": {
                "name": "Openness to experience of the essay writer",
                "label_key": "personality_openess",
                "template": WASSA2023EvalTemplate(
                    name="writer_openness",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    input="\nThe score for the Openness to experience of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_extraversion": {
                "name": "Extraversion of the essay writer",
                "label_key": "personality_extraversion",
                "template": WASSA2023EvalTemplate(
                    name="writer_openness",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    input="\nThe score for the Extraversion of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_agreeableness": {
                "name": "Agreeableness of the essay writer",
                "label_key": "personality_agreeableness",
                "template": WASSA2023EvalTemplate(
                    name="writer_agreeableness",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    input="\nThe score for the Agreeableness of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_stability": {
                "name": "Stability of the essay writer",
                "label_key": "personality_stability",
                "template": WASSA2023EvalTemplate(
                    name="writer_agreeableness",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    input="\nThe score for the Stability of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_perspective_taking": {
                "name": "Perspective-taking of the essay writer",
                "label_key": "iri_perspective_taking",
                "template": WASSA2023EvalTemplate(
                    name="writer_perspective_taking",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    input="\nThe score for the Perspective-taking of the essay writer is ",
                ),
                "category": "IRI"
            },
            "writer_personal_distress": {
                "name": "Personal distress of the essay writer",
                "label_key": "iri_personal_distress",
                "template": WASSA2023EvalTemplate(
                    name="writer_personal_distress",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    input="\nThe score for the Personal distress of the essay writer is ",
                ),
                "category": "IRI"
            },
            "writer_fantasy": {
                "name": "Fantasy of the essay writer",
                "label_key": "iri_fantasy",
                "template": WASSA2023EvalTemplate(
                    name="writer_fantasy",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    input="\nThe score for the Fantasy of the essay writer is ",
                ),
                "category": "IRI"
            },
            "writer_empathic_concern": {
                "name": "Empathic Concern of the essay writer",
                "label_key": "iri_empathatic_concern",
                "template": WASSA2023EvalTemplate(
                    name="writer_empatheic_concern",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    input="\nThe score for the Empathic Concern of the essay writer is ",
                ),
                "category": "IRI"
            }
        }

    @property
    def task_mapping(self):
        return task_mapping

    @classmethod
    def parse_example(
            cls,
            template,
            example: Dict[str, str],
            label_key,
            dataset_name
    ) -> Tuple[str, str]:
        article = "\n[Article]\n{article}\n[End of Article]".format(article=example['article'])
        return_json = f"""
    Provide your evaluation in JSON format, as shown in the example below.

    Example of Evaluation Output:
    ```json
    {{"{label_key}": 0.8}}
    ```
            """
        label = example[label_key]
        if dataset_name == 'conversation':
            conversation = "\n[Conversation]\n {history}\n[End of Conversation]".format(history=example['history'])
            response = "\n[Response by Speaker {speaker_id}] {text} [End of Response by Speaker {speaker_id}]".format(
                speaker_id=example['speaker_id'],
                text=example['text'])
            if isinstance(label, float):
                label = "{:.2f}".format(label)
            return "".join([article] + [conversation] + [response] + [template.input] + [return_json]), label
        else:
            essay = "\n[Essay by Speaker {speaker_id}]\n{essay}\n[End of Essay by Speaker {speaker_id}]".format(
                speaker_id=example['speaker_id'],
                essay=example['essay'])
            if isinstance(label, float):
                label = "{:.2f}".format(label)
            return "\n\n".join([article] + [essay] + [template.input] + [return_json]), label

    def run(self, split, n_shot, output_dir, num_retries=5):
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
                    parse_func=self.parse_example
                )

                response = self.prompt_for_response(messages, num_retries)

                outputs[category].setdefault(label_key, [])
                outputs[category][label_key].append(response[label_key])


@register_evaluator("wassa2023", "multi_scorer")
class WASSA2023MultiScorerEvaluator(WASSA2023Evaluator):

    def __init__(self,
                 task_dir,
                 model_name,
                 model_api_key,
                 model_api_base,
                 evaluator_name,
                 evaluator_api_key,
                 evaluator_api_base):
        super().__init__(task_dir,
                         model_name, model_api_key, model_api_base, evaluator_name, evaluator_api_key,
                         evaluator_api_base)

    @property
    def categories(self):
        return {
            "CONV": {
                "name": "at the speech-turn-level",
                "label_key": ["EmotionalPolarity", "Emotion", "Empathy"],
                "template": WASSA2023EvalTemplate(
                    name="conversation",
                    instruction="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict:\n"
                                "1. EmotionalPolarity: as a score from the range : 0 to 2 ;\n"
                                "2. Emotion: as a score from the range : 1 to 5 ;\n"
                                "3. Empathy: as a score from the range : 1 to 5 .\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {{\"EmotionalPolarity\": 1.3, \"Emotion\": 3.6, \"Empathy\": 4.3}}\n"
                          "```",
                ),
                "category": "CONV"
            },
            "EMP": {
                "name": "Empathy at the essay-level",
                "label_key": ["empathy", "distress"],
                "template": WASSA2023EvalTemplate(
                    name="essay",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Empathy score is an average of 7-point scale ratings, representing each of the following states (warm, tender, sympathetic,softhearted, moved, compassionate). Try to predict:\n"
                                "1. empathy as a score from the range : 1 to 7:\n"
                                "2. distress as a score from the range : 1 to 7:\n"
                                "\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {{\"empathy\": 5.8, \"distress\": 2.5}}\n"
                          "```",
                ),
                "category": "EMP"
            },
            "EMO": {
                "name": "Emotion at the essay-level",
                "label_key": "emotion",
                "template": WASSA2023EvalTemplate(
                    name="essay_emotion",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} from one or more emotion labels from the Ekman’s six basic emotions (sadness, joy, disgust, surprise, anger, or fear) as well as neutral. The essay expresses the emotion:\n\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {{\"emotion\": \"disgust\"}}\n"
                          "```",
                ),
                "category": "EMO"
            },
            "PER": {
                "name": "essay writer",
                "label_key": [
                    "conscientiousness",
                    "openess",
                    "extraversion",
                    "agreeableness",
                    "stability"
                ],
                "template": WASSA2023EvalTemplate(
                    name="writer_conscientiousness",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict metrics of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n"
                                "1. conscientiousness:\n"
                                "2. openness\n"
                                "3. extraversion\n"
                                "4. agreeableness\n"
                                "5. stability\n"
                                "\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {{\"conscientiousness\": 1.3, \"openess\": 3.6, \"extraversion\": 4.3, \"agreeableness\": 3.6, \"stability\": 4.3}}\n"
                          "```",
                ),
                "category": "PER"
            },
            "IRI": {
                "name": "Perspective-taking of the essay writer",
                "label_key": [
                    "perspective_taking",
                    "personal_distress",
                    "fantasy",
                    "empathatic_concern"
                ],
                "template": WASSA2023EvalTemplate(
                    name="writer_perspective_taking",
                    instruction="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict metrics of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n"
                                "1. perspective_taking\n"
                                "2. personal_distress\n"
                                "3. fantasy\n"
                                "4. empathatic_concern\n"
                                "\n",
                    input="Provide your evaluation in JSON format, as shown in the example below.\n"
                          "Example of Evaluation Output:\n"
                          "```json\n"
                          "  {{\"perspective_taking\": 1.3, \"personal_distress\": 3.6, \"fantasy\": 4.3, \"empathatic_concern\": 3.6}}\n"
                          "```",
                ),
                "category": "IRI"
            }
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
            query, resp = "\n\n".join([article] + [conversation] + [response] + [template.input]), json.dumps(label)
        else:
            essay = "[Essay by Speaker {speaker_id}]\n{essay}\n[End of Essay by Speaker {speaker_id}]".format(
                speaker_id=example['speaker_id'],
                essay=example['essay'])
            query, resp = "\n\n".join([article] + [essay] + [template.input]), json.dumps(label)
        logger.debug(query)
        logger.debug(resp)
        return query, resp

    def run(self, split, n_shot, output_dir, num_retries=5):

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

            with open(os.path.join(output_dir, "res", f"predictions_{category}.tsv"), "w") as fd:
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

                    fd.write("\t".join(map(lambda x: "{:.2f}".format(x), result)) + "\n")

        if split == "validation":
            os.makedirs(os.path.join(output_dir, "ref"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "res"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "ret"), exist_ok=True)

            with zipfile.ZipFile(os.path.join(self.task_dir, self.task, f'{self.task}.zip')) as zd:
                for filename in zd.namelist():
                    if filename in ['goldstandard_dev.tsv', 'goldstandard_CONV_dev.tsv']:
                        with open(os.path.join(output_dir, "ref", filename.replace("_dev", "")), 'wb') as fd:
                            with zd.open(filename) as f:
                                fd.write(f.read())
            score(output_dir, os.path.join(output_dir, 'ret'))
