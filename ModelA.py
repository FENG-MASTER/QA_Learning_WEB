from CommonModel import CommonModel


class ModelA(CommonModel):
    def _getAnswerFeature(self, answer):
        feature = [answer['answer_id'], answer['question_id'], answer['comment_count'], answer['voteup_count'],
                   answer['create_time'], answer['update_time'], answer['len']]

        ans_tags = self.db.get_answer_key_word_info_by_id(answer['answer_id'])

        feature_key_word = [0 for i in range(len(self.search_ques_key_word_ids))]

        for t in ans_tags:
            if t[0] in self.search_ques_key_word_ids:
                feature_key_word[self.search_ques_key_word_ids.index(t[0])] = t[1] + 1

        feature.extend(feature_key_word)
        return feature

    def _createQuestionFeatureMap(self):

        question_id_key_word_weight_score_map = {}

        _key_word_index = 0
        for wid in self.search_ques_key_word_ids:
            # 把相关问题的信息记录下来
            o = self.db.get_question_by_key_word_id(wid)
            for content in o:
                # +1是防止小于1出现相乘越来越小的情况
                _question_id = int(content[0])
                _weight = content[1] + 1

                if _question_id not in question_id_key_word_weight_score_map:
                    _empty_map = [1]
                    _empty_map.extend([0 for i in range(self.question_tag_len)])
                    question_id_key_word_weight_score_map[_question_id] = _empty_map

                _o_level = question_id_key_word_weight_score_map[_question_id][0]
                _o_weight_list = question_id_key_word_weight_score_map[_question_id][1:]

                _o_weight_list[_key_word_index] = _weight + _o_weight_list[_key_word_index]+1

                _res=[_o_level + 1]
                _res.extend(_o_weight_list)
                question_id_key_word_weight_score_map[_question_id] =_res

            _key_word_index += 1

        self.question_id_key_word_weight_score_map=question_id_key_word_weight_score_map

    def _getQuestionFeature(self, question):

        if not hasattr(self,"question_id_key_word_weight_score_map"):
            self._createQuestionFeatureMap()

        if int(question["question_id"]) in self.question_id_key_word_weight_score_map:
            return self.question_id_key_word_weight_score_map[int(question["question_id"])]
        else:
            return [0 for i in range(self.question_tag_len+1)]

