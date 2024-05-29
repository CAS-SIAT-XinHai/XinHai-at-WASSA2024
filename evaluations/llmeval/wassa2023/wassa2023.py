# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datasets
import pandas as pd

_CITATION = """\
@inproceedings{barriere-etal-2023-findings,
    title = "Findings of {WASSA} 2023 Shared Task on Empathy, Emotion and Personality Detection in Conversation and Reactions to News Articles",
    author = "Barriere, Valentin  and
      Sedoc, Jo{\~a}o  and
      Tafreshi, Shabnam  and
      Giorgi, Salvatore",
    editor = "Barnes, Jeremy  and
      De Clercq, Orph{\'e}e  and
      Klinger, Roman",
    booktitle = "Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, {\&} Social Media Analysis",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wassa-1.44",
    doi = "10.18653/v1/2023.wassa-1.44",
    pages = "511--525",
    abstract = "This paper presents the results of the WASSA 2023 shared task on predicting empathy, emotion, and personality in conversations and reactions to news articles. Participating teams were given access to a new dataset from Omitaomu et al. (2022) comprising empathic and emotional reactions to news articles. The dataset included formal and informal text, self-report data, and third-party annotations. Specifically, the dataset contained news articles (where harm is done to a person, group, or other) and crowd-sourced essays written in reaction to the article. After reacting via essays, crowd workers engaged in conversations about the news articles. Finally, the crowd workers self-reported their empathic concern and distress, personality (using the Big Five), and multi-dimensional empathy (via the Interpersonal Reactivity Index). A third-party annotated both the conversational turns (for empathy, emotion polarity, and emotion intensity) and essays (for multi-label emotions). Thus, the dataset contained outcomes (self-reported or third-party annotated) at the turn level (within conversations) and the essay level. Participation was encouraged in five tracks: (i) predicting turn-level empathy, emotion polarity, and emotion intensity in conversations, (ii) predicting state empathy and distress scores, (iii) predicting emotion categories, (iv) predicting personality, and (v) predicting multi-dimensional trait empathy. In total, 21 teams participated in the shared task. We summarize the methods and resources used by the participating teams.",
}

"""

_DESCRIPTION = """\
Participants are given a new dataset of empathic reactions to news stories and associated conversations which contains essays in reaction to news articles where there is harm to a person, group, or other (from Omitaomu et al. 2022, similar to Buechel et al. 2018). 
Those essays contain Batson empathic concern and personal distress scores, as well as the Big Five personality and Inter-Personal Index (IRI) scores of each user. 
This new dataset also contains conversations between two users that read the same article. 
Each of their speech turn has been annotated in perceived empathy, emotion polarity and emotion intensity. 
The essays are between 300 and 800 characters in length. 
The conversations contain in average 23 speech turns. 
The dataset also includes the news articles and person-level demographic information (age, gender, ethnicity, income, education level).

Track 1: Empathy and Emotion Prediction in Conversations (CONV), which consists in predicting the perceived empathy, emotion polarity and emotion intensity at the speech-turn-level in a conversation
Track 2: Empathy Prediction (EMP), which consists in predicting both the empathy concern and the personal distress at the essay-level
Track 3: Emotion Classification (EMO), which consists in predicting the emotion at the essay-level
Track 4: Personality Prediction (PER), which consists in predicting the personality of the essay writer, knowing all his/her essays and the news article from which they reacted
Track 5: Interpersonal Reactivity Index Prediction (IRI), which consists in predicting the personality of the essay writer, knowing all his/her essays and the news article from which they reacted
"""

_HOMEPAGE = "https://codalab.lisn.upsaclay.fr/competitions/11167"

_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"

_URL = "wassa2023.zip"

task_list = [
    "conversation",
    "essay"
]

question_types = {'打分题'}


class WASSA2023Config(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class WASSA2023(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        WASSA2023Config(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        if self.config.name == 'conversation':
            features = datasets.Features(
                {
                    "conversation_id": datasets.Value("int32"),
                    "turn_id": datasets.Value("int32"),
                    "history": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "speaker_number": datasets.Value("string"),
                    "article_id": datasets.Value("int32"),
                    "speaker_id": datasets.Value("int32"),
                    "essay_id": datasets.Value("int32"),
                    "article": datasets.Value("string"),
                    "EmotionalPolarity": datasets.Value("float32"),
                    "Emotion": datasets.Value("float32"),
                    "Empathy": datasets.Value("float32"),
                }
            )
        else:
            features = datasets.Features(
                {
                    "split": datasets.Value("string"),
                    "conversation_id": datasets.Value("int32"),
                    "article_id": datasets.Value("int32"),
                    "article": datasets.Value("string"),
                    "essay_id": datasets.Value("int32"),
                    "essay": datasets.Value("string"),
                    "empathy": datasets.Value("float32"),
                    "emotion": datasets.Value("string"),
                    "distress": datasets.Value("float32"),
                    "speaker_number": datasets.Value("int32"),
                    "speaker_id": datasets.Value("int32"),
                    "gender": datasets.Value("string"),
                    "education": datasets.Value("string"),
                    "race": datasets.Value("string"),
                    "age": datasets.Value("string"),
                    "income": datasets.Value("string"),
                    "personality_conscientiousness": datasets.Value("float32"),
                    "personality_openness": datasets.Value("float32"),
                    "personality_extraversion": datasets.Value("float32"),
                    "personality_agreeableness": datasets.Value("float32"),
                    "personality_stability": datasets.Value("float32"),
                    "iri_perspective_taking": datasets.Value("float32"),
                    "iri_personal_distress": datasets.Value("float32"),
                    "iri_fantasy": datasets.Value("float32"),
                    "iri_empathetic_concern": datasets.Value("float32"),
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name

        df_articles = pd.read_csv(f"{data_dir}/articles_adobe_AMT.csv",
                                  encoding='utf-8')
        df_articles.rename(columns={'text': 'article'}, inplace=True)
        if task_name == 'conversation':
            train_file = f"{data_dir}/WASSA23_conv_level_with_labels_train.tsv"
            validation_file = f"{data_dir}/WASSA23_conv_level_dev.tsv"
            validation_labels_file = f'{data_dir}/goldstandard_CONV_dev.tsv'
            test_file = f"{data_dir}/WASSA23_conv_level_test.tsv"
            labels_names = [
                "EmotionalPolarity",
                "Emotion",
                "Empathy"
            ]
        elif task_name == 'essay':
            train_file = f"{data_dir}/WASSA23_essay_level_with_labels_train.tsv"
            validation_file = f"{data_dir}/WASSA23_essay_level_dev.tsv"
            validation_labels_file = f'{data_dir}/goldstandard_dev.tsv'
            test_file = f"{data_dir}/WASSA23_essay_level_test.tsv"
            labels_names = [
                "empathy",
                "distress",
                "emotion",
                "personality_conscientiousness",
                "personality_openness",
                "personality_extraversion",
                "personality_agreeableness",
                "personality_stability",
                "iri_perspective_taking",
                "iri_personal_distress",
                "iri_fantasy",
                "iri_empathetic_concern"
            ]
        else:
            raise ValueError

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_file,
                    "df_articles": df_articles
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": validation_file,
                    "df_articles": df_articles,
                    "labels_file": validation_labels_file,
                    "labels_names": labels_names
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_file,
                    "df_articles": df_articles
                },
            )
        ]

    def _generate_examples(self, filepath, df_articles, labels_file=None, labels_names=None):
        df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
        if labels_file and labels_names:
            df_labels = pd.read_table(labels_file, names=labels_names)
            df = pd.concat([df, df_labels], axis=1)

        merged_df = pd.merge(df, df_articles, on='article_id', how='inner')
        if self.config.name == 'conversation':
            merged_df.sort_values(['conversation_id', 'turn_id'], inplace=True)
            label_columns = ['EmotionalPolarity', 'Emotion', 'Empathy']
            history = {}
            for conversation_id, group_df in merged_df.groupby('conversation_id'):
                history[conversation_id] = [f"Speaker {int(s)}: {t}" for s, t in
                                            zip(group_df.speaker_id.tolist(), group_df.text.tolist())]

            for i, instance in enumerate(merged_df.to_dict(orient="records")):
                for k in label_columns:
                    if k not in instance:
                        instance[k] = 0
                instance['history'] = "\n".join(history[instance['conversation_id']][:instance['turn_id']])
                yield i, instance
        else:
            label_columns = ["empathy", "distress", "emotion", "personality_conscientiousness",
                             "personality_openness", "personality_extraversion",
                             "personality_agreeableness",
                             "personality_stability", "iri_perspective_taking",
                             "iri_personal_distress",
                             "iri_fantasy", "iri_empathetic_concern"]

            for i, instance in enumerate(merged_df.to_dict(orient="records")):
                for k in label_columns:
                    if k not in instance or instance[k] == 'unknown':
                        instance[k] = 0
                yield i, instance
