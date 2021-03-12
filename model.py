import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from load_data import GetSquadDL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging

def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

class LitEncoder(pl.LightningModule):
    def __init__(self, embedding_size=768, gru_hidden_size=768, latent_size=768):
        super().__init__()
        
        self.gru = nn.GRU(embedding_size, gru_hidden_size, num_layers=1, dropout=0.5, bidirectional=False)

        # Attention weights. 
        self.l1 = nn.Linear(gru_hidden_size, 256)
        self.l1.bias.data.fill_(0)
        self.l2 = nn.Linear(256, 10)
        self.l2.bias.data.fill_(0)

        self.softmax = nn.Softmax(dim=0)
        self.l_mu = nn.Linear(gru_hidden_size, latent_size)
        self.l_var = nn.Linear(gru_hidden_size, latent_size)
        self.tanh = nn.Tanh()

        self.apply(weights_init)
        
    def forward(self, x):
        #x = pack_padded_sequence(x)

        h_t, _ = self.gru(x) # seq x batch x 768

        #h_t, _ = pad_packed_sequence(h_t)

        a = self.tanh(self.l1(h_t)) # seq x batch x 256
        a = self.l2(a) # seq x batch x 10
        a = self.softmax(a) 
        #print(h_t.shape[1], torch.sum(a, 0).sum()) ## should be equal to the batch size x 10.    
        a = a.permute(1, 0, 2).transpose(1, 2)
        h_t = h_t.permute(1, 0, 2)
        m = torch.sum(a @ h_t, 1).squeeze(1) # batch x 10 x seq * batch x seq x 768 => batch x 10 x 768 => batch x 1 x 768 => batch x 768
        #The matrix multiplication(s) are done between the last two dimensions (1×8 @ 8×16 --> 1×16). 
        #The remaining first three dimensions are broadcast and are ‘batch’, so you get 10×64×1152 matrix multiplications.
        #m = torch.sum(a * h_t, 0) 

        mu = self.l_mu(m)
        log_var = self.l_var(m)
        
        return mu, log_var

class LitDecoder(pl.LightningModule):
    def __init__(self, vocab_size, embedding_size=768, gru_hidden_size=768):
        super().__init__()
        
        self.gru_cell = nn.GRUCell(embedding_size, gru_hidden_size)
        self.l_out = nn.Linear(gru_hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, z):
        h = self.gru_cell(x, z)
        o = self.l_out(h)
        o = self.softmax(o)
        #print(torch.sum(o, 0).sum())
        return o, h

class LitCrossVAE(pl.LightningModule):
    def __init__(self, bert, vocab_size, h_params, ds):
        super().__init__()

        if ds != None:
            ds_path, tokenizer, batch_size = ds
            self.train_dl, self.val_dl = GetSquadDL(ds_path, tokenizer, batch_size=batch_size)

        self.bert = bert
        
        self.E_a = LitEncoder()
        self.E_q = LitEncoder()
        
        self.D_a = LitDecoder(vocab_size)
        self.D_q = LitDecoder(vocab_size)

        self.vocab_size = vocab_size
        
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

        self.alpha, self.beta, self.gamma = h_params

        self.param_a = list(self.E_a.parameters()) + list(self.D_a.parameters())
        self.param_q = list(self.E_q.parameters()) + list(self.D_q.parameters())

        # self.matching_loss = 0
        # self.reconstruction_loss = 0
        # self.total_loss = 0
        
    def forward(self, q, a, case=''):
        #print(q.shape, a.shape)
        q = self.get_bert_embedding(q).detach()

        # getting the latent spaces.
        # the latent space will be 1 embedding for the entire sentence. 
        # q shape: words * batch * embedding(768)
        # q = torch.randn(32, 4, 768)
        mu_q, log_var_q = self.E_q(q)

        # sampling z_a from the above space.
        z_q = self.sample(mu_q, log_var_q)

        if (case == 'z'):
            return z_q

        a = self.get_bert_embedding(a).detach()
        # a = torch.randn(64, 4, 768)
        #print('embeddings ', q.shape, a.shape)

        mu_a, log_var_a = self.E_a(a)
        #print('encoders (mean and var) ', mu_a.shape, log_var_a.shape, mu_q.shape, log_var_q.shape)
        
        # sampling z_a from the above space.
        z_a = self.sample(mu_a, log_var_a)
        #print('samples ', z_a.shape, z_q.shape)
        
        sim = self.similarity(z_a, z_q)
        #print('sim ', sim.shape)
        
        if (case == 'sim'):
            return sim
        else:
            # decoding the latent space. 
            q_hat = self.generate(z_a, q, self.D_q)
            a_hat = self.generate(z_q, a, self.D_a)
            #print('decoder ', q_hat.shape, a_hat.shape)

            return z_a, mu_a, log_var_a, z_q, mu_q, log_var_q, a_hat, q_hat, sim
        
    def generate(self, z, target, dec):
        target_len = len(target)
        target_hat = torch.zeros(target_len, target.shape[1], self.vocab_size, device=self.device) #batch_size, self.vocab_size)
        inp = target[0]
        
        for i in range(target_len):
            out, z = dec(inp, z)
            
            target_hat[i] = out
            best_guess = out.argmax(1).view(-1, 1, 1) # 0th dimension is batch size, 1st dimension is word embedding
            #print('best guess - ', best_guess.shape)
            inp = target[i] if random.random() < 0.5 else self.get_bert_embedding(best_guess).squeeze(0)
            #print('decoder instance ', out.shape, z.shape, inp.shape)
        
        #print('generated shape ', target_hat.shape)
        return target_hat
    
    def similarity(self, z_a, z_q):
        return self.cos_sim(z_a, z_q).unsqueeze(1)
    
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mu, std)
        return dist.rsample()

    def get_bert_embedding(self, sentence):
        #return self.bert(sentence.squeeze(1)).last_hidden_state.permute(1, 0, 2)
        # self.wordEmbedding = nn.Embedding.from_pretrained(args.pretrain).to(self.device)
        # self.wordEmbedding.weight.requires_grad = True
        return self.bert.embeddings.word_embeddings(sentence.squeeze(1)).permute(1, 0, 2)

    def train_dataloader(self):
        return self.train_dl

    def training_step(self, batch, batch_idx, optimizer_idx):
        # training_step defined the train loop.
        # It is independent of forward
        #q, p_ans, n_ans = batch

        # optim_a, optim_q = self.optimizers()
        
        q, p_ans = batch

        # Positive pair. 
        z_a, mu_a, log_var_a, z_q, mu_q, log_var_q, a_hat, q_hat, sim = self.forward(q, p_ans)
        vae = z_a, mu_a, log_var_a, z_q, mu_q, log_var_q, a_hat, q_hat, p_ans, q
        #print(len(vae))
        p_loss = self.get_loss(sim, 1, vae)
        #print('p_loss ', p_loss)

        n_ans1 = p_ans[torch.randperm(p_ans.size()[0])]
        n_ans2 = p_ans[torch.randperm(p_ans.size()[0])]

        # Negative pair.
        sim1 = self.forward(q, n_ans1, case='sim')
        sim2 = self.forward(q, n_ans2, case='sim')

        n_loss1 = self.get_loss(sim1, 0)
        n_loss2 = self.get_loss(sim2, 0)

        #loss = p_loss + n_loss1 + n_loss2

        # self.manual_backward(loss, optim_a, retain_graph=True)
        # self.manual_backward(loss, optim_q)
        # loss.backward()

        # optim_a.step()
        # optim_q.step()

        # optim_a.zero_grad()
        # optim_q.zero_grad()

        return p_loss + n_loss1 + n_loss2

    # def training_epoch_end(self, ouputs):
        # self.beta = min(self.beta*2, 1)
        # logging.info(f'total_loss: {self.total_loss} reconstruction_loss: {self.reconstruction_loss} matching_loss: {self.matching_loss}')
        # self.total_loss, self.reconstruction_loss, self.matching_loss = 0, 0, 0

    def val_dataloader(self):
        return self.val_dl

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            q, p_ans = batch
            #print(q.shape, p_ans.shape)

            sim_pos = self.forward(q, p_ans, case='sim')

            n_ans1 = p_ans[torch.randperm(p_ans.size()[0])]
            n_ans2 = p_ans[torch.randperm(p_ans.size()[0])]

            sim_n1 = self.forward(q, n_ans1, case='sim')
            sim_n2 = self.forward(q, n_ans2, case='sim')

            loss = self.get_loss(sim_pos, 1) + self.get_loss(sim_n1, 0) + self.get_loss(sim_n2, 0) 
            return {'val_loss': loss}

    def configure_optimizers(self):
        optim_a = torch.optim.Adam(self.param_a, lr=1e-4, weight_decay=1e-8, amsgrad=True)
        optim_q = torch.optim.Adam(self.param_q, lr=1e-4, weight_decay=1e-8, amsgrad=True)
        return optim_a, optim_q

    def get_loss(self, sim, y, vae=None):
        ## loss-matching
        #print(sim.shape)
        #print(sim, torch.ones_like(sim)*y)
        #loss_matching = self.bce(sim, torch.ones_like(sim)*y) ## changing as per the comment from the author
        loss_matching = 0

        #if self.current_epoch > 5: 
        loss_matching = self.mse(sim, torch.ones_like(sim)*y)

        #print('loss_matching ', loss_matching)
        
        if y == 1 and vae != None:
            z_a, mu_a, log_var_a, z_q, mu_q, log_var_q, a_hat, q_hat, a, q = vae
            ## loss-cross
            #print(q.permute(2, 0, 1).shape, q_hat.shape)
            q = q.permute(2, 0, 1).reshape(-1)
            a = a.permute(2, 0, 1).reshape(-1)

            q_hat = q_hat.reshape(-1, q_hat.shape[2])
            a_hat = a_hat.reshape(-1, a_hat.shape[2])

            #print(q_hat.shape, q.shape, a_hat.shape, a.shape)

            ## Probably doesn't work for text this way. Not sure, might try later. Trying CE for now.
        #     p_rec_q = torch.distributions.Normal(q_hat, torch.ones_like(q_hat)) # q_hat reconstructed from z_a 
        #     p_rec_a = torch.distributions.Normal(a_hat, torch.ones_like(a_hat)) # a_hat reconstructed from z_q

        #     log_pqza = p_rec_q.log_prob(q)
        #     log_pazq = p_rec_a.log_prob(a)

        #     loss_cross = -1 * y * (log_pqza + log_pazq)
            q = q.to(self.device)
            #print(q.device, q_hat.device, sim.device)

            loss_cross = self.ce(q_hat, q) + self.ce(a_hat, a)
            #loss_cross = loss_cross * 0.0
            
            #print('loss_cross ', loss_cross)
        
            ## loss-kl
            # std_a = torch.exp(0.5 * log_var_a)
            # std_q = torch.exp(0.5 * log_var_q)

            # #print(std_a.shape, std_q.shape)

            # p_n = torch.distributions.Normal(torch.zeros_like(mu_a), torch.ones_like(log_var_a))
            # q_za = torch.distributions.Normal(mu_a, std_a)
            # q_zq = torch.distributions.Normal(mu_q, std_q)

            # log_qza_a = q_za.log_prob(z_a)
            # log_qzq_q = q_zq.log_prob(z_q)
            # log_pza = p_n.log_prob(z_a)
            # log_pzq = p_n.log_prob(z_q)
            
            # #print(log_qza_a, log_pza, log_qzq_q, log_pzq)

            # loss_kl = ((log_qza_a - log_pza) + (log_qzq_q - log_pzq))
            # #print(loss_kl.shape)
            # loss_kl = loss_kl.sum(-1)
            # #print(loss_kl)
            # #print(loss_kl.shape)
            # loss_kl = loss_kl.mean()
            # print('loss_kl ', loss_kl)
            kl_q = torch.mean(-0.5 * torch.sum(1 + log_var_q - mu_q ** 2 - log_var_q.exp(), dim = 1), dim = 0)
            kl_a = torch.mean(-0.5 * torch.sum(1 + log_var_a - mu_a ** 2 - log_var_a.exp(), dim = 1), dim = 0)

            factor = 0.1
            f2 = 2.0 * factor * (self.current_epoch / 100)
            beta_auto = max(f2, factor)
        
            return self.gamma * loss_matching + self.alpha * (loss_cross + beta_auto * (kl_a + kl_q))

        return self.gamma * loss_matching