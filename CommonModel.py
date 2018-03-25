import jieba
from sklearn.externals import joblib
from numpy import *
import numpy as np

from DBHelper import DBHelper


class CommonModel(object):
    def __init__(self,model_save_path, question_tag_len=10,question_feature_len=11):
        self.question_tag_len = question_tag_len
        self.question_feature_len=question_feature_len
        self.model=joblib.load(model_save_path)


    def getBestAnswerList(self, question_title):
        self.question_title = question_title
        self.db = DBHelper.get_instance()
        tags = jieba.analyse.extract_tags(question_title, topK=self.question_tag_len, withWeight=True,
                                          allowPOS=['ns', 'n', 'vn', 'v', 'nr'])

        # 问题关键词ID列表
        self.search_ques_key_word_ids = [self.db.get_key_word_id(t[0]) for t in tags]
        self.search_ques_key_word_weights = [t[1] for t in tags]

        while len(self.search_ques_key_word_ids) < 10:
            self.search_ques_key_word_ids.append(0)
            self.search_ques_key_word_weights.append(0)

        answer_list = self._getAboutAnswer()
        question_list = self._getAboutQuestion()

        # 特征值列表,每个数组第一位表示答案ID
        featureX = []

        for ans in answer_list:
            afeature = self._getAnswerFeature(ans)
            if afeature[1] in question_list:
                #             答案回答的问题在相关问题中,
                _question_feature = self._getQuestionFeature(afeature[1])
                _f = [afeature[0], afeature[2:]]
                _f.extend(_question_feature[2:])
                featureX.append(_f)
            else:
                _question_feature = [0 for i in range(self.question_feature_len)]
                _f = [afeature[0], afeature[2:]]
                _f.extend(_question_feature)
                featureX.append(_f)

        featureX=array(featureX)
        featureX=self._data_preprocess(featureX)

        scores=self.model.predict(featureX[1:])

        score_list = np.vstack((array(featureX[:0]), array(scores)))

        sl = list(score_list.transpose().tolist())

        sl.sort(key=lambda d: d[1], reverse=True)

        return sl

    def _data_preprocess(self,data):
        return data

    def _getAboutQuestion(self):
        """
        获取相关问题
        :return: 相关问题
        """

        a_question_list = []
        for wid in self.search_ques_key_word_ids:
            # 把相关问题的信息记录下来
            o = self.db.get_question_by_key_word_id(wid)
            for content in o:
                # +1是防止小于1出现相乘越来越小的情况
                _question_id = content[0]
                a_question_list.append(_question_id)

        a_question_list = set(a_question_list)

        question_list = {}

        for qid in a_question_list:
            question_list[int(qid)] = self.db.get_question_by_id(qid)

        return question_list

    def _getAboutAnswer(self):
        """
        获取相关答案
        :return: 相关答案
        """
        ans_id_list = []

        for qid in self.search_ques_key_word_ids:
            # 有关键词
            o = self.db.get_answers_by_key_word_id(qid)
            for content in o:
                ans_id_list.append(content[0])

        ans_id_list = set(ans_id_list)

        ans_list = []

        for ans_id in ans_id_list:
            ans_list.append(self.db.get_answer_by_id(ans_id))

        return ans_list

    def _getAnswerFeature(self, ans):
        """
        获取答案特征
        :param ans:
        :return:
                返回一个数组, 第一个位置表示答案ID,第二个位置表示回答的问题ID

        """
        feature = []
        return feature

    def _getQuestionFeature(self, question):
        """
        获取问题特征
        :param question:
        :return:
            返回一个数组,第一个位置表示问题ID
        """
        feature = []
        return feature

    def _predict(self, answer_id_list):
        pass
