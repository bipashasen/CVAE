import re
import json 
import collections
from random import randrange
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from ReQA import GenerateExamples, QuestionAnswerInput, IndexedItem, SquadDataset
from config import MAX_SEQ_LEN_Q, MAX_SEQ_LEN_A, dl_num_workers 

class SquadCustomDataset():
    def __init__(self, path, tokenizer, case):
        super(SquadCustomDataset, self).__init__()

        self.case_ans = 'a'
        self.case_ques = 'q'
        self.tokenizer = tokenizer
        self.sentence_tokens = collections.defaultdict(None)
        self.case = case
        loaded_squad = self._load_data(path)
        
        self.queries = loaded_squad.queries
        self.master_idx = loaded_squad.master_index
        self.response_index = loaded_squad.response_index

        self.response_idx_keys = list(self.response_index.keys())
        
    def __getitem__(self, idx):
        if self.case == self.case_ans:
            return self._get_answers(idx)
        elif self.case == self.case_ques:
            return self._get_question(idx)
        else:
            return self._get_train_pair(idx)

    def __len__(self):
        return len(self.response_index) if self.case == self.case_ans else len(self.queries) 

    def _get_train_pair(self, idx):
        query = self.queries[idx]
        
        q_token = self.sentence_tokens[query.query]
        pos_a = query.response
        pos_a_token = self.sentence_tokens[pos_a]

        return [q_token, pos_a_token]
        
        # num_ans = len(self.master_idx)
        # rand_idx = randrange(num_ans)

        # pos_a_idx = self.response_index.get(pos_a)
        
        # while(rand_idx in pos_a_idx):
        #     rand_idx = randrange(num_ans)
            
        # neg_a = self.master_idx[rand_idx].sentence
        # neg_a_token = self.sentence_tokens[neg_a]
        
        # return [q_token, pos_a_token, neg_a_token]

    def _get_question(self, idx):
        query = self.queries[idx]

        return [query.query, query.response, self.sentence_tokens[query.query]]

    def _get_answers(self, idx):
        m_i_idx = self.response_index[self.response_idx_keys[idx]][0]

        ans = self.master_idx[m_i_idx].sentence
        
        return [ans, self.sentence_tokens[ans]]

    def _load_data(self, path):
        with open(path) as f:
            squad_json = json.load(f)

        questions = set()
        qa_count = 0
        queries = [] # type: List[QuestionAnswerInput]
        
        master_index = []  # type: List[IndexedItem]
        seen_responses = set()  # type: Set[Tuple[str, str]]
        for question, answer, document, paragraph in GenerateExamples(squad_json): 
            ## this is getting the question, answer, corresponding paragraph, and all sentences in paragraph
            questions.add(question)
            queries.append(QuestionAnswerInput(question, answer, document, paragraph.id, None))
            qa_count += 1
            self.get_tokenized(question, 'q')
            
            # all sentences in a paragraph, making sentence-parahgraph pairs. candidate answers. 
            for sentence in paragraph.sentences:
                if (sentence, document) not in seen_responses:
                    seen_responses.add((sentence, document))
                    self.get_tokenized(sentence)
                    master_index.append(IndexedItem(sentence, document, None, paragraph.id))
                    #logging.info("questions=%s, QA inputs=%s, index_size=%s", len(questions), qa_count, len(master_index))

        response_index = collections.defaultdict(list)  # type: Dict[str, List[int]]
        for i, (sentence, _, _, _) in enumerate(master_index):
            response_index[sentence].append(i)

        #print('total dataset loaded with ', qa_count, ' qeuestions')
        return SquadDataset(queries, master_index, response_index)     

    def get_tokenized(self, sentence, case='a'):
        if sentence not in self.sentence_tokens.keys():
            max_length = MAX_SEQ_LEN_Q if case == 'q' else MAX_SEQ_LEN_A
            self.sentence_tokens[sentence] = self.tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True).input_ids

def GetSquadDL(squad_path, bert_tokenizer, batch_size, case=''):
    squadCustomDataset = SquadCustomDataset(squad_path, bert_tokenizer, case)

    if case == '':
        tot_count = len(squadCustomDataset)
        train_count = int(0.9*tot_count)
        val_count = tot_count-train_count
        #print(tot_count, train_count, val_count)

        train_dataset, valid_dataset, test_dataset = random_split(
            squadCustomDataset, (train_count, val_count, 0)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dl_num_workers)
        validation_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=dl_num_workers)

        return train_loader, validation_loader

    return DataLoader(dataset = squadCustomDataset, batch_size=batch_size, num_workers=dl_num_workers)
