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

    def _getQuestionFeature(self, question):
        feature=[]


        pass
