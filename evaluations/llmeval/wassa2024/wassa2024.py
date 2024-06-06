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
"""

_DESCRIPTION = """\
Task Description
Participants are given a new dataset of empathic reactions to news stories and associated conversations which contains essays in reaction to news articles where there is harm to a person, group, or other (from Omitaomu and Tafreshi et al. 2023, similar to Buechel et al. 2018). Those essays contain Batson empathic concern and personal distress scores, as well as the Big Five personality (OCEAN) and Inter-Personal Index (IRI) scores of each user. This new dataset also contains conversations between two users that read the same article. Each of their speech turn has been annotated in perceived empathy, emotion polarity, and emotion intensity. The essays are between 300 and 800 characters in length. The conversations contains 11,788 speech turns. The dataset also includes the news articles and person-level demographic information (age, gender, ethnicity, income, education level).

You can participate in five different tracks:

Track 1: Empathy Prediction in Conversations (CONV-dialog), which consists in predicting the perceived empathy at the dialog-level
Track 2: Empathy and Emotion Prediction in Conversations Turns (CONV-turn), which consists in predicting the perceived empathy, emotion polarity, and emotion intensity at the speech-turn-level in a conversation
Track 3: Empathy Prediction (EMP), which consists in predicting both the empathy concern and the personal distress at the essay-level
Track 4: Personality Prediction (PER), which consists in predicting the personality (openness, conscientiousness, extraversion, agreeableness, and emotional stability; OCEAN) of the essay writer, knowing all his/her essays, dialogs, and the news article from which they reacted
"""

_HOMEPAGE = "https://codalab.lisn.upsaclay.fr/competitions/18810"

_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"

_URL = "wassa2024.zip"

task_list = [
    "CONVD",
    "CONVT",
    "EMP",
    "PER"
]

question_types = {'打分题'}

gender_map = {1: "Male", 2: "Female", 5: "Other"}
race_map = {
    1: "White",
    2: "Hispanic / Latino",
    3: "Black / African American",
    4: "Native American / American Indian",
    5: "Asian / Pacific Islander",
    6: "Other"
}
education_map = {
    1: "Less than a high school diploma",
    2: "High school degree or diploma",
    3: "Technical / Vocational School",
    4: "Some college",
    5: "Two year associate degree",
    6: "College or university degree",
    7: "Postgraduate / professional degree"
}


class WASSA2024Config(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class WASSA2024(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        WASSA2024Config(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        if self.config.name == 'CONVD':
            # id, conversation_id, article_id, "person_id_1", this_persons_perceived_empathy_of_other_person
            # id, article_id, "title", "source", "text", "objectOfSuffering"

            features = datasets.Features(
                {
                    "conversation_id": datasets.Value("int32"),
                    "history": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "person_id_1": datasets.Value("string"),
                    "article_id": datasets.Value("int32"),
                    "title": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "article": datasets.Value("string"),
                    "Empathy": datasets.Value("float32"),
                }
            )
        elif self.config.name == 'CONVT':
            # id, article_id, conversation_id, turn_id, "speaker", "text", "person_id_1", "person_id_2", Emotion, EmotionalPolarity, Empathy, SelfDisclosure
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

        elif self.config.name == "EMP":
            # conversation_id, article_id, "person_id", "person_essay", person_empathy, person_distress

            features = datasets.Features(
                {
                    "conversation_id": datasets.Value("int32"),
                    "history": datasets.Value("string"),
                    "article_id": datasets.Value("int32"),
                    "article": datasets.Value("string"),
                    "person_essay": datasets.Value("string"),
                    "person_empathy": datasets.Value("float32"),
                    "person_distress": datasets.Value("float32"),
                }
            )
        elif self.config.name == "PER":
            # conversation_id, article_id, "person_id", "person_essay", person_empathy, person_distress

            features = datasets.Features(
                {
                    "person_id": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                    "education": datasets.Value("string"),
                    "race": datasets.Value("string"),
                    "age": datasets.Value("string"),
                    "income": datasets.Value("string"),
                    "perceived_empathy": datasets.Value("string"),
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
        else:
            raise NotImplementedError

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)

        df_articles_train = pd.read_csv(f"{data_dir}/ec_article_info_train.csv",
                                        escapechar='\\')
        df_articles_train.rename(columns={'text': 'article'}, inplace=True)

        df_articles_dev = pd.read_csv(f"{data_dir}/ec_article_info_dev.csv",
                                      escapechar='\\')
        df_articles_dev.rename(columns={'text': 'article'}, inplace=True)

        df_conversations_dialogue_train = pd.read_csv(f"{data_dir}/trac1_CONVD_train.csv", escapechar='\\')
        df_conversations_dialogue_train.rename(columns={'person_id_1': 'person_id'}, inplace=True)
        df_conversations_dialogue_dev = pd.read_csv(f"{data_dir}/trac1_CONVD_dev.csv", escapechar='\\')

        df_conversations_turn_train = pd.read_csv(f"{data_dir}/trac2_CONVT_train.csv",
                                                  escapechar='\\')
        df_conversations_turn_dev = pd.read_csv(f"{data_dir}/trac2_CONVT_dev.csv",
                                                escapechar='\\')
        df_emp_train = pd.read_csv(f"{data_dir}/trac3_EMP_train.csv", escapechar='\\')
        df_emp_dev = pd.read_csv(f"{data_dir}/trac3_EMP_dev.csv", escapechar='\\')

        df_per_train = pd.read_csv(f"{data_dir}/trac4_PER_train.csv", escapechar='\\')
        df_per_dev = pd.read_csv(f"{data_dir}/trac4_PER_dev.csv", escapechar='\\')

        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "filepath": test_file,
            #         "df_articles": df_articles_dev
            #     },
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "df_articles": df_articles_dev,
                    "df_conversations_dialogue": df_conversations_dialogue_dev,
                    "df_conversations_turn": df_conversations_turn_dev,
                    "df_emp": df_emp_dev,
                    "df_per": df_per_dev,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "df_articles": df_articles_train,
                    "df_conversations_dialogue": df_conversations_dialogue_train,
                    "df_conversations_turn": df_conversations_turn_train,
                    "df_emp": df_emp_train,
                    "df_per": df_per_train,
                },
            )
        ]

    def _generate_examples(self,
                           df_articles,
                           df_conversations_dialogue,
                           df_conversations_turn,
                           df_emp,
                           df_per):
        if self.config.name == 'CONVD':
            merged_df = pd.merge(df_conversations_dialogue, df_articles, on='article_id', how='inner')
            df_conversations_turn.sort_values(['conversation_id', 'turn_id'], inplace=True)
            history = {}
            for conversation_id, group_df in df_conversations_turn.groupby('conversation_id'):
                if "speaker" in group_df.columns:
                    history[conversation_id] = [f"{s}: {t}" for s, t in
                                                zip(group_df.speaker.tolist(), group_df.text.tolist())]
                else:
                    history[conversation_id] = [f"Speaker {int(s)}: {t}" for s, t in
                                                zip(group_df.speaker_id.tolist(), group_df.text.tolist())]
            for i, instance in enumerate(merged_df.to_dict(orient="records")):
                try:
                    instance['history'] = "\n".join(history[instance['conversation_id']])
                except KeyError:
                    print(f"Conversation {instance['conversation_id']} has no history")
                    instance['history'] = ""
                yield i, instance
        elif self.config.name == 'CONVT':
            df_conversations_turn.sort_values(['conversation_id', 'turn_id'], inplace=True)
            merged_df = pd.merge(df_conversations_turn, df_articles, on='article_id', how='inner')
            history = {}
            for conversation_id, group_df in df_conversations_turn.groupby('conversation_id'):
                if "speaker" in group_df.columns:
                    history[conversation_id] = [f"{s}: {t}" for s, t in
                                                zip(group_df.speaker.tolist(), group_df.text.tolist())]
                else:
                    history[conversation_id] = [f"Speaker {int(s)}: {t}" for s, t in
                                                zip(group_df.speaker_id.tolist(), group_df.text.tolist())]
            for i, instance in enumerate(merged_df.to_dict(orient="records")):
                try:
                    instance['history'] = "\n".join(history[instance['conversation_id']][:instance['turn_id'] - 1])
                except KeyError:
                    print(f"Conversation {instance['conversation_id']} has no history")
                    instance['history'] = ""
                yield i, instance
        elif self.config.name == 'EMP':
            merged_df = pd.merge(df_emp, df_articles, on='article_id', how='inner')
            df_conversations_turn.sort_values(['conversation_id', 'turn_id'], inplace=True)
            history = {}
            for conversation_id, group_df in df_conversations_turn.groupby('conversation_id'):
                if "speaker" in group_df.columns:
                    history[conversation_id] = [f"{s}: {t}" for s, t in
                                                zip(group_df.speaker.tolist(), group_df.text.tolist())]
                else:
                    history[conversation_id] = [f"Speaker {int(s)}: {t}" for s, t in
                                                zip(group_df.speaker_id.tolist(), group_df.text.tolist())]
            for i, instance in enumerate(merged_df.to_dict(orient="records")):
                try:
                    instance['history'] = "\n".join(history[instance['conversation_id']])
                except KeyError:
                    print(f"Conversation {instance['conversation_id']} has no history")
                    instance['history'] = ""
                yield i, instance
        elif self.config.name == 'PER':
            df_conversations_turn.sort_values(['conversation_id', 'turn_id'], inplace=True)
            history = {}
            for conversation_id, group_df in df_conversations_turn.groupby('conversation_id'):
                if "speaker" in group_df.columns:
                    history[conversation_id] = [f"{s}: {t}" for s, t in
                                                zip(group_df.speaker.tolist(), group_df.text.tolist())]
                else:
                    history[conversation_id] = [f"Speaker {int(s)}: {t}" for s, t in
                                                zip(group_df.speaker_id.tolist(), group_df.text.tolist())]

            dialogue = {}
            merged_df = pd.merge(df_conversations_dialogue, df_articles, on='article_id', how='inner')
            for i, instance in enumerate(merged_df.to_dict(orient="records")):
                try:
                    instance['history'] = "\n".join(history[instance['conversation_id']])
                except KeyError:
                    print(f"Conversation {instance['conversation_id']} has no history")
                    instance['history'] = ""
                dialogue.setdefault(instance['person_id'], [])
                dialogue[instance['person_id']].append(instance)

            for i, instance in enumerate(df_per.to_dict(orient="records")):
                try:
                    instance['perceived_empathy'] = dialogue[instance['person_id']]
                except KeyError:
                    print(f"Person {instance['person_id']} has no records perceiving empathy of other person!")

                instance['personality_openness'] = instance.pop("personality_openess")
                instance['iri_empathetic_concern'] = instance.pop("iri_empathatic_concern")
                yield i, instance
