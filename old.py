





from imports import *
from config import mode, batch_size



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device in use ', device)


#print('vocab_size ', tokenizer.vocab_size)

# Whether the model returns all hidden-states.
# Load in eval mode.


















from imports import *
from model import CrossVAE, CrossVAELoss
from load_data import _load_data, get_dataloader
from config import alpha, beta, gamma, store_path, n_gpu, load_model_from_path

n_epochs = 100
lr=0.00001
p_epochs = 5
e_epochs = 1000
train_eval_q_samples = 200
candidate_samples = 100
anneal_alpha = 1
anneal_beta = 1

cvae = nn.DataParallel(CrossVAE())
cvae = cvae.to(device)
optim = torch.optim.Adam(cvae.parameters(), lr=lr)

load = load_model_from_path

model_path = store_path

if load == True:  
    checkpoint = torch.load(model_path)
    cvae.load_state_dict(checkpoint['cvae'])

    torch.save(checkpoint, model_path + str(time.time()))

def load_dev_data():
    dev_path = squad_dev_path
    with open(dev_path) as f:
        _dev_squad = json.load(f)

    _dev_loaded_data = _load_data(_dev_squad)
    #print(len(_dev_loaded_data.queries), len(_dev_loaded_data.master_index))

    return _dev_loaded_data

def model_training(t_alpha, t_beta, t_gamma):
    #model_evaluation()
    dataloader = get_dataloader(squad_train_path)
    dev_data = load_dev_data()
    cvae.train()

    print("training ..", flush=True)
    for epoch in range(n_epochs):
        tot_n_loss = 0
        tot_p_loss = 0

        # as mentioned by the author, turning it off.
        # if epoch % anneal_alpha == 0:
        #     t_alpha = min(t_alpha*2, 1)

        if epoch % anneal_beta == 0:
            t_beta = min(t_beta*2, 1)

        for i, (q, p_ans, n_ans) in enumerate(dataloader):
            

            tot_p_loss += p_loss.item()
            tot_n_loss += n_loss.item()

            if i % p_epochs == 0:
                print(f'epoch: {epoch}/{n_epochs}, steps: {i}/{len(dataloader)}, c_p_loss: {p_loss.item():.2f}, p_loss: {(tot_p_loss/(i+1)):.2f}, c_n_loss: {n_loss.item():.2f}, n_loss {(tot_n_loss/(i+1)):.2f}', flush=True)

                torch.save({
                    'cvae': cvae.state_dict()
                }, model_path)

            if i % e_epochs == 0:
                model_evaluation(dev_data, 'train')
                cvae.train()

def print_mean_metrics(mean_rank, tp_1, tp_5, mrr, q_len):
    print(f'mean_rank: {(mean_rank/q_len):.2f}, mean_tp_1: {(tp_1/q_len):.2f}, mean_tp_5: {(tp_5/q_len):.2f}, mrr: {(mrr/q_len):.2f}', flush=True)


        

model_training(alpha, beta, gamma)
#model_evaluation()
#model_evaluation(None, 'train')



def model_evaluation(_dev_loaded_data=None, mode='test'):
    cvae.eval()
    # model evaluation
    if _dev_loaded_data is None:
        _dev_loaded_data = load_dev_data()

    q_len = len(_dev_loaded_data.queries)
    ans_len = len(_dev_loaded_data.response_index)

    bert = bert_model.to(device)

    response_index = _dev_loaded_data.response_index
    answers = list(response_index.keys())
    a_tokens = bert_tokens(answers, 'a').to(device)
    #a_tokens = [lambda x: _dev_loaded_data.tokens[x] for x in answers]
    #print(a_tokens.shape)

    #a_list = a_list.view(1, a_list.shape[0], a_list.shape[1])
    #print(a_list.shape)

    q_i = -1
    q_len = len(_dev_loaded_data.queries)
    tp_1 = 0
    tp_5 = 0
    mean_rank = 0
    n_samples = candidate_samples
    topk = 5
    mrr = 0

    eval_batch = n_gpu*10

    print("evaluating .. ", flush=True)
    with torch.no_grad():
        for query in _dev_loaded_data.queries:
            q_i = q_i + 1
            if q_i % 50 == 0: 
                print(q_i, flush=True)
            
            question = query.query
            response = query.response
            response_idx = answers.index(response)

            #q_token = bert_tokens(question).to(device)
            q_token = _dev_loaded_data.tokens[question].to(device)
            #print(q_token.shape)

            idx_samples = torch.randperm(n_samples)
            a_tokens_sampled = a_tokens[idx_samples]

            if response_idx not in idx_samples:
                replace_idx = random.randrange(0, n_samples)
                a_tokens_sampled[replace_idx] = a_tokens[response_idx]
                idx_samples[replace_idx] = response_idx

            y = torch.zeros(a_tokens_sampled.shape[0])
            #print(a_tokens_sampled.shape)

            a_i = -1

            for i in range(int(n_samples/eval_batch)):
                start_idx = i*eval_batch
                end_idx = start_idx + eval_batch
                a_tokens_in = a_tokens_sampled[start_idx:end_idx].unsqueeze(1)
                q_tokens_in = q_token.repeat(1, eval_batch).view(eval_batch, 1, -1)
                #print(a_tokens_in.shape, q_tokens_in.shape)

                #print(a_tokens_in.shape, q_tokens_in.shape)
                out = cvae(q_tokens_in, a_tokens_in, only_sim=True).squeeze(1)
                y[start_idx:end_idx] = out

            # for a_token in a_tokens_sampled:
            #     a_i = a_i + 1

            #     a_token = a_token.view(1, 1, -1).to(device)
            #     q_token = q_token.view(1, 1, -1).to(device)
            #     print(a_token.shape, q_token.shape)
            #     y[a_i] = cvae(q_token, a_token, only_sim=True)

            #print(y)

            #print(answers)
            np_answers = np.array(answers)
            #print(np_answers[0])

            _, idx_top = torch.topk(y, topk)
            top_sen = answers[idx_samples[idx_top[0]]]
            #print(top_sen)
            top5_sen = np_answers[idx_samples[idx_top]]
            #print(top5_sen)

            if top_sen == response:
                tp_1 = tp_1 + 1
                tp_5 = tp_5 + 1
                rank = 0
            elif response in top5_sen:
                tp_5 = tp_5 + 1
                rank = (top5_sen == response).nonzero()[0].item()
            else:
                _, idx_sorted = torch.topk(y, n_samples)
                rank = idx_sorted.tolist().index(idx_samples.tolist().index(response_idx))
                #print(rank)

            mrr = mrr + (1/(rank+1))
            mean_rank = mean_rank + rank + 1

            #print(rank, tp_1, tp_5)
            if mode == 'train' and q_i == train_eval_q_samples:
                print_mean_metrics(mean_rank, tp_1, tp_5, mrr, (q_i+1))
                return
            
        print_mean_metrics(mean_rank, tp_1, tp_5, mrr, q_len)