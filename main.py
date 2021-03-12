from load_data import GetSquadDL
from model import LitCrossVAE
from transformers import BertTokenizer, BertModel
import torch
import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from config import n_gpu, n_nodes, mode, alpha, beta, gamma, batch_size, task

mode = 'debug'
task = 'train'

if mode == 'debug':
    squad_train_path = '/home2/bipasha31/squad/test-v1.1.json'
    squad_dev_path = squad_train_path
else:
    squad_train_path = '/home2/bipasha31/squad/train-v1.1.json'
    squad_dev_path = '/home2/bipasha31/squad/dev-v1.1.json'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)

# authors reploaded corpus of vocab and tokenizer, and bert embedding. 
# preloaded_dir = '/home2/bipasha31/python_scripts/cvae_ar/auth_code/CVAE/utilities/data/'

for param in bert.parameters():
    param.requires_grad = False 

vocab_size = tokenizer.vocab_size

#print(vocab_size)

h_params = alpha, beta, gamma

if task == 'train':
    cvae = LitCrossVAE(bert=bert, vocab_size=vocab_size, h_params=h_params, ds=[squad_train_path, tokenizer, batch_size])

    saveTopK_callback = ModelCheckpoint(
        #filepath=os.getcwd(),
        monitor='val_loss',
        save_top_k=5,
        verbose=True,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=n_gpu, 
        profiler="simple",
        #resume_from_checkpoint='/home2/bipasha31/python_scripts/cvae_ar/lightning_logs/version_340385/checkpoints/epoch=10-step=4520.ckpt',
        #fast_dev_run=True,
        max_epochs=60,
        check_val_every_n_epoch=1,
        #automatic_optimization=False,
        #val_percent_check=0.2,
        #num_nodes=n_nodes, 
        #precision=16, 
        checkpoint_callback=saveTopK_callback,
        accelerator='ddp')

    trainer.fit(cvae)

elif task == 'test':
    
    #m_path = '/home2/bipasha31/python_scripts/cvae_ar/lightning_logs/version_340384/checkpoints/epoch=3-step=1643.ckpt'
    # mean_rank: 5141.568359375, ranks_1: 0, ranks_5: 6, gt: 10570
    # m_path = '/home2/bipasha31/python_scripts/cvae_ar/lightning_logs/version_340385/checkpoints/epoch=10-step=4520.ckpt'
    # mean_rank: 5175.84814453125, ranks_1: 3, ranks_5: 10, gt: 10570
    #m_path = '/home2/bipasha31/python_scripts/cvae_ar/lightning_logs/version_340388/checkpoints/epoch=12-step=5342.ckpt'
    # mean_rank: 5083.0703125, ranks_1: 1, ranks_5: 4, gt: 10570
    m_path = "/home2/bipasha31/python_scripts/cvae_ar/lightning_logs/version_340389/checkpoints/epoch=17-step=3563.ckpt"
    m_path = "/home2/bipasha31/python_scripts/cvae_ar/lightning_logs/version_340389/checkpoints/epoch=29-step=5939.ckpt"

    cvae = LitCrossVAE.load_from_checkpoint(m_path, bert=bert, vocab_size=vocab_size, h_params=h_params, ds=None).to('cuda')

    squadDL_a = GetSquadDL(squad_dev_path, tokenizer, batch_size=batch_size*4, case='a')
    squadDL_q = GetSquadDL(squad_dev_path, tokenizer, batch_size=batch_size*4, case='q')

    ans = []
    ques = []
    q_ans = []

    encodings_a = torch.empty(1, 512, device='cuda')
    encodings_q = torch.empty(1, 512, device='cuda')

    cvae.eval()

    with torch.no_grad():
        for i, (a, a_token) in enumerate(squadDL_a):
            a_token = a_token.to('cuda')
            ans.append(a)
            encodings_a = torch.cat((encodings_a, cvae(a_token, a=None, case='z')), dim=0)

        ans = [item for sublist in ans for item in sublist]

        for i, (q, a, q_token) in enumerate(squadDL_q):
            q_token = q_token.to('cuda')
            ques.append(q)

            a_idx = [ans.index(item) for item in a]

            q_ans.append(a_idx)
            encodings_q = torch.cat((encodings_q, cvae(q_token, a=None, case='z')), dim=0)

        q_ans = [item for sublist in q_ans for item in sublist]
    
    #print(f'encoding_q: {encodings_q.shape}, encoding_a: {encodings_a.shape}')

    encodings_a = encodings_a[1:]
    encodings_q = encodings_q[1:]

    print(f'encoding_q: {encodings_q.shape}, encoding_a: {encodings_a.shape}')

    encodings_a_norm = encodings_a / encodings_a.norm(dim=1)[:, None]
    encodings_q_norm = encodings_q / encodings_q.norm(dim=1)[:, None]

    cos_sim = torch.mm(encodings_q_norm, encodings_a_norm.transpose(0,1))

    _, sim_idx = torch.sort(cos_sim, dim=1)
    q_ans = torch.tensor(q_ans, device='cuda').reshape(-1, 1)
    ranks = ((q_ans == sim_idx).nonzero(as_tuple=True)[1]).float()
    ranks_reciprocal = torch.reciprocal(ranks)
    ranks_1 = (ranks == 1).sum()
    ranks_lt6 = (ranks < 6).sum()
    den = len(q_ans)

    print(f'cos_sim: {cos_sim.shape}, sim_idx: {sim_idx.shape}, q_ans: {q_ans.shape}, ranks: {ranks.shape}')

    print(f'mean_rank: {torch.mean(ranks)}, ranks_1: {int(ranks_1)}, ranks_5: {int(ranks_lt6)}, gt: {int(den)}')
    print(f'mean_rank_reciprocal: {torch.mean(ranks_reciprocal)}, recall@1: {(ranks_1/den):.3f}, recall@5: {(ranks_lt6/den):.3f}')
