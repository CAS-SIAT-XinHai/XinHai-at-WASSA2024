import os
import os.path
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

from math import sqrt

from . import register_evaluator
from .base import BaseEvaluator

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


@register_evaluator("wassa2024", "baseline")
class WASSA2024Evaluator(BaseEvaluator):

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

    @property
    def categories(self):
        return {
            "conversation_emotion_polarity": {
                "name": "Emotion Polarity at the speech-turn-level",
                "label_key": "EmotionalPolarity",
                "template": WASSA2024EvalTemplate(
                    name="conversation_emotion_polarity",
                    system="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} as a score from the range : 0 to 2:\n\n",
                    instruction="\n The score for the response on emotion polarity is ",
                ),
                "category": "CONV"
            },
            "conversation_emotion_intensity": {
                "name": "Emotion Intensity at the speech-turn-level",
                "label_key": "Emotion",
                "template": WASSA2024EvalTemplate(
                    name="conversation_emotion_intensity",
                    system="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} as a score from the range : 1 to 5:\n\n",
                    instruction="\n The score for the response on emotion intensity is ",
                ),
                "category": "CONV"
            },
            "conversation_empathy": {
                "name": "Empathy at the speech-turn-level",
                "label_key": "Empathy",
                "template": WASSA2024EvalTemplate(
                    name="conversation_empathy",
                    system="Read the conversation between speakers in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} as a score from the range : 1 to 5:\n\n",
                    instruction="\n The score for the response on empathy is ",
                ),
                "category": "CONV"
            },
            "essay_empathy": {
                "name": "Empathy at the essay-level",
                "label_key": "empathy",
                "template": WASSA2024EvalTemplate(
                    name="essay_empathy",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Empathy score is an average of 7-point scale ratings, representing each of the following states (warm, tender, sympathetic,softhearted, moved, compassionate). Try to predict {subject} as a score from the range : 1 to 7:\n\n",
                    instruction="\n The score for the response on empathy is ",
                ),
                "category": "EMP"
            },
            "essay_distress": {
                "name": "Personal Distress at the essay-level",
                "label_key": "distress",
                "template": WASSA2024EvalTemplate(
                    name="essay_distress",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Distress score is an average of 7-point scale ratings, representing each of the following states (worried, upset, troubled, perturbed, grieved, disturbed, alarmed,distressed). Try to predict {subject} as a score from the range : 1 to 7:\n\n",
                    instruction="\n The score for the response on distress is ",
                ),
                "category": "EMP"
            },
            "essay_emotion": {
                "name": "Emotion at the essay-level",
                "label_key": "emotion",
                "template": WASSA2024EvalTemplate(
                    name="essay_emotion",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} from one or more emotion labels from the Ekman’s six basic emotions (sadness, joy, disgust, surprise, anger, or fear) as well as neutral:\n\n",
                    instruction="\nThe essay expresses the emotion ",
                ),
                "category": "EMO"
            },
            "writer_conscientiousness": {
                "name": "Conscientiousness of the essay writer",
                "label_key": "personality_conscientiousness",
                "template": WASSA2024EvalTemplate(
                    name="writer_conscientiousness",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    instruction="\nThe score for the Conscientiousness of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_openness": {
                "name": "Openness to experience of the essay writer",
                "label_key": "personality_openess",
                "template": WASSA2024EvalTemplate(
                    name="writer_openness",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    instruction="\nThe score for the Openness to experience of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_extraversion": {
                "name": "Extraversion of the essay writer",
                "label_key": "personality_extraversion",
                "template": WASSA2024EvalTemplate(
                    name="writer_openness",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    instruction="\nThe score for the Extraversion of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_agreeableness": {
                "name": "Agreeableness of the essay writer",
                "label_key": "personality_agreeableness",
                "template": WASSA2024EvalTemplate(
                    name="writer_agreeableness",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    instruction="\nThe score for the Agreeableness of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_stability": {
                "name": "Stability of the essay writer",
                "label_key": "personality_stability",
                "template": WASSA2024EvalTemplate(
                    name="writer_agreeableness",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Big 5 personality traits ( also known as the OCEAN model ) as a score from the range : 1 to 7:\n\n",
                    instruction="\nThe score for the Stability of the essay writer is ",
                ),
                "category": "PER"
            },
            "writer_perspective_taking": {
                "name": "Perspective-taking of the essay writer",
                "label_key": "iri_perspective_taking",
                "template": WASSA2024EvalTemplate(
                    name="writer_perspective_taking",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    instruction="\nThe score for the Perspective-taking of the essay writer is ",
                ),
                "category": "IRI"
            },
            "writer_personal_distress": {
                "name": "Personal distress of the essay writer",
                "label_key": "iri_personal_distress",
                "template": WASSA2024EvalTemplate(
                    name="writer_personal_distress",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    instruction="\nThe score for the Personal distress of the essay writer is ",
                ),
                "category": "IRI"
            },
            "writer_fantasy": {
                "name": "Fantasy of the essay writer",
                "label_key": "iri_fantasy",
                "template": WASSA2024EvalTemplate(
                    name="writer_fantasy",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    instruction="\nThe score for the Fantasy of the essay writer is ",
                ),
                "category": "IRI"
            },
            "writer_empathic_concern": {
                "name": "Empathic Concern of the essay writer",
                "label_key": "iri_empathatic_concern",
                "template": WASSA2024EvalTemplate(
                    name="writer_empatheic_concern",
                    system="Read the essay written by a speaker in reaction to a news article where there is harm to a person, group, or other. Try to predict {subject} of the Interpersonal Reactivity Index (IRI), a measurement tool for the multidimensional assessment of empathy, as a score from the range : 1 to 5:\n\n",
                    instruction="\nThe score for the Empathic Concern of the essay writer is ",
                ),
                "category": "IRI"
            }
        }

    @property
    def task_mapping(self):
        return task_mapping
