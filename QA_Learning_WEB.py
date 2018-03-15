from flask import Flask
from flask import render_template
from AnswerModel import AnswerModel
from flask import request
import pickle
import time
import jieba
import jieba.analyse
from DBHelper import DBHelper
from sklearn import preprocessing
from numpy import *
import numpy as np
from sklearn.externals import joblib

from  sklearn.linear_model import ElasticNet

app = Flask(__name__)

model_path = 'F:\py\QA_Learning_WEB\qa_model.m'
qa_model = None
db = DBHelper.get_instance()


@app.route('/')
def index():
    return render_template('Search.html')


@app.route('/answer/<int:answer_id>')
def answer(answer_id):
    ans = db.get_answer_by_id(answer_id)
    ans['update_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ans['update_time']))
    ans['create_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ans['create_time']))
    ques = db.get_question_by_id(str(ans['question_id']))
    return render_template('Answer_detail.html', answer=ans, question=ques)


@app.route('/result/', methods=['GET', 'POST'])
def result():
    answer_id_list = get_all_answer_list(request.form['question'])
    if answer_id_list is None or len(answer_id_list)==0:
        return render_template('NoFound.html',search_question=request.form['question'])

    answer_list = get_answers_by_ids(answer_id_list[0:100])

    return render_template('Result.html', resultList=answer_list)


def get_answers_by_ids(id_list):
    result = list()
    for item in id_list:
        ans = db.get_answer_by_id(int(item[0]))
        ans['update_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ans['update_time']))
        ans['question_title'] = db.get_question_by_id(str(ans['question_id']))['title']
        result.append(ans)
    return result


def get_all_answer_list(question):
    tags = jieba.analyse.extract_tags(question, topK=8, withWeight=True,allowPOS=['ns', 'n', 'vn', 'v','nr'])

    # 问题关键词ID列表
    ques_key_word_ids = [db.get_key_word_id(t[0]) for t in tags]
    ques_key_word_weights = [t[1] for t in tags]

    ans_id_list = []

    for qid in ques_key_word_ids:
        # 有关键词
        o = db.get_answers_by_key_word_id(qid)
        for content in o:
            ans_id_list.append(content[0])

    # list_all_ans=[]
    # for wid in ques_key_word_ids:
    #     o = db.get_answers_by_key_word_id(wid)
    #     list_all_ans.append([content[0] for content in o])

    # for ans_l in list_all_ans:
    #     if len(list_all_ans)==0:
    #         ans_id_list=ans_l
    #     else:
    #         t_ans_id_list=list(set(ans_l).intersection(set(ans_id_list)))
    #         if len(t_ans_id_list)==0:
    #             break


    for need in range(8 - len(ques_key_word_ids)):
        ques_key_word_ids.append(0)
        ques_key_word_weights.append(0)

    ans_id_list = set(ans_id_list)

    # 包含关键词的所有问题列表(或)

    new_question_id_key_word_weight_score_map = {}

    _key_word_index = 0
    for wid in ques_key_word_ids:
        # 把相关问题的信息记录下来
        o = db.get_question_by_key_word_id(wid)
        for content in o:
            # +1是防止小于1出现相乘越来越小的情况
            _question_id = content[0]
            _weight = content[1] + 1

            if _question_id not in new_question_id_key_word_weight_score_map:
                new_question_id_key_word_weight_score_map[_question_id] = [1, [1, 1, 1, 1, 1, 1, 1, 1], 0]

            _o_level = new_question_id_key_word_weight_score_map[_question_id][0]
            _o_weight_list = new_question_id_key_word_weight_score_map[_question_id][1]
            _o_weight_list[_key_word_index] = _weight + _o_weight_list[_key_word_index]

            new_question_id_key_word_weight_score_map[_question_id] = [_o_level + 1, _o_weight_list, 0]

        _key_word_index += 1

    if len(new_question_id_key_word_weight_score_map)!=0:

        for c in new_question_id_key_word_weight_score_map:
            # 计算问题相关度的分数
            item = new_question_id_key_word_weight_score_map[c]
            wl = item[1]
            _score = 0
            for s in wl:
                _score += s
            item[2] = _score * item[0]
            new_question_id_key_word_weight_score_map[c] = item

        new_question_id_list = []
        new_question_level_list = []
        new_question_score_list = []
        new_question_weight_list = []

        for qis in new_question_id_key_word_weight_score_map:
            new_question_id_list.append(qis)
            new_question_level_list.append(new_question_id_key_word_weight_score_map[qis][0])
            new_question_weight_list.append(new_question_id_key_word_weight_score_map[qis][1])
            new_question_score_list.append(new_question_id_key_word_weight_score_map[qis][2])

        new_question_score_list = preprocessing.MinMaxScaler().fit_transform(
            array(new_question_score_list).reshape(-1, 1)).tolist()
        # new_question_weight_list = preprocessing.MinMaxScaler().fit_transform(array(new_question_weight_list)).tolist()

        new_question_id_key_word_weight_score_map.clear()

        for _index in range(len(new_question_id_list)):
            _qid = int(new_question_id_list[_index])
            _level = new_question_level_list[_index]
            _weight = new_question_weight_list[_index]
            _score = new_question_score_list[_index][0]
            new_question_id_key_word_weight_score_map[_qid] = [_level, _weight, _score]


        # -----------------提取含有问题关键词的问题列表-----------------#

    score_list = []

    test_ALL_X = []

    for _id in ans_id_list:

        ans = db.get_answer_by_id(_id)
        key_word_feature = []
        feature = []

        # 数据库获取答案关键词特征值(可能不够15个,需要处理)
        ans_key_word_info = db.get_answer_key_word_info_by_id(_id)

        # 如果关键词特征值不够15个,则后面补0
        if len(ans_key_word_info) < 15:
            for i in range(15 - len(ans_key_word_info)):
                ans_key_word_info.append([0, 0])

        # 答案关键词ID列表
        ans_key_word_ids = [_info[0] for _info in ans_key_word_info]

        _i = 0

        for i in range(len(ques_key_word_ids)):
            if ques_key_word_ids[i] in ans_key_word_ids:
                # 如果某关键词ID在问题关键词和答案关键词中都有,那么命中,计算特征值(用相应权值相乘
                key_word_feature.append(
                    ques_key_word_weights[i] * (
                        (ans_key_word_info[ans_key_word_ids.index(ques_key_word_ids[i])][1]) + 1))
                _i += 1
            else:
                # 如果没有,直接置为0
                key_word_feature.append(0)

        cy = 0
        for _i in key_word_feature:
            if _i != 0:
                cy += 1

        feature = [ans['create_time'], ans['update_time'], ans['comment_count'], ans['len'], cy]
        feature.extend(key_word_feature)

        # 问题相似度的分数计算

        _ans_ques_id = ans['question_id']

        if _ans_ques_id in new_question_id_key_word_weight_score_map:
            # 如果这个问题回答的是相关问题的,则有加分

            # 这个问题所回答的问题和当前问题关键词的相似情况
            feature.extend(new_question_id_key_word_weight_score_map[_ans_ques_id][1])

            # 这个问题所回答的问题和当前问题的关键字相同数目
            feature.append(new_question_id_key_word_weight_score_map[_ans_ques_id][0])

        else:
            for _i in range(9):
                feature.append(0)

        test_ALL_X.append(feature)

    if len(test_ALL_X)==0:
        return None

    test_arr_x = array(test_ALL_X)
    test_arr_x = np.hsplit(test_arr_x, (13,))



    s = qa_model.predict(np.hstack((preprocessing.MinMaxScaler().fit_transform(test_arr_x[0]), test_arr_x[1])))
    score_list = np.vstack((array(list(ans_id_list)), array(s)))

    sl = list(score_list.transpose().tolist())

    sl.sort(key=lambda d: d[1], reverse=True)

    return sl


if __name__ == '__main__':
    qa_model = joblib.load(model_path)
    app.run()
