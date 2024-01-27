import torch
from networks.modules import *
from networks.st2m_trainers import st2m_TrainerV13
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm

from scripts.slerp import do_slerp_op
import os
import codecs as cs


def build_models_tran2_clip(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    tran_inferencer = TRANSITION_TRANSFORMER_2code(max_mov_len=opt.max_motion_length // opt.unit_length,
                                                   device=opt.device)

    his_text_encoder = TextEncoderCLIP(hidden_size=opt.dim_dec_hidden,
                                       device=opt.device)

    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, tran_inferencer, his_text_encoder


class st2mV13GeneratedDatasetV2_reallen(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, tran_infer, his_text_enc = build_models_tran2_clip(opt)
        trainer = st2m_TrainerV13(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, tran_infer, his_text_enc,
                                 mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)


        min_mov_length = 4
        # print(mm_idxs)

        print('Loading model: Epoch %03d' % (epoch))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb_0, word_emb_1, pos_ohot_0, pos_ohot_1, caption_0, caption_1, \
                cap_lens_0, cap_lens_1, motions_0, motions_1, m_lens_0, m_lens_1, tokens_0, tokens_1, \
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens, id_name = data

                tokens_0 = tokens_0[0].split('_')
                tokens_1 = tokens_1[0].split('_')
                tokens = tokens[0].split('_')


                word_emb_0 = word_emb_0.detach().to(opt.device).float()
                word_emb_1 = word_emb_1.detach().to(opt.device).float()
                pos_ohot_0 = pos_ohot_0.detach().to(opt.device).float()
                pos_ohot_1 = pos_ohot_1.detach().to(opt.device).float()



                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                # if is_mm:
                #     print(mm_num_now, i, mm_idxs[mm_num_now])
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                # print(m_lens[0].item(), cap_lens[0].item())
                for t in range(repeat_times):
                    mov_len_0 = torch.div(m_lens_0[0], opt.unit_length, rounding_mode='trunc')
                    his_movements = trainer.mov_enc(
                        torch.zeros((word_emb_0.shape[0], opt.unit_length, opt.dim_pose - 4),
                                    device=opt.device)
                    ).repeat(1, mov_len_0, 1)  # (bs, mov_len, mov_dim)
                    his_real_mov_lens = torch.tensor(1).repeat(word_emb_0.shape[0]).to(opt.device)
                    his_caption = ['start'] * word_emb_0.shape[0]

                    pred_motions_0, _, _, his_movements, his_real_mov_lens = trainer.generate_transition_2code(
                        word_emb_0,
                        pos_ohot_0,
                        cap_lens_0,
                        m_lens_0,
                        mov_len_0,
                        opt.dim_pose,
                        his_movements,
                        his_real_mov_lens,
                        his_caption)

                    mov_len_1 = torch.div(m_lens_1[0], opt.unit_length, rounding_mode='trunc')
                    his_caption = [caption_0[0]]

                    pred_motions_1, _, _, his_movements, his_real_mov_lens = trainer.generate_transition_2code(
                        word_emb_1,
                        pos_ohot_1,
                        cap_lens_1,
                        m_lens_1,
                        mov_len_1,
                        opt.dim_pose,
                        his_movements,
                        his_real_mov_lens,
                        his_caption)

                    dict_motion = np.concatenate([pred_motions_0[0].cpu().numpy(), pred_motions_1[0].cpu().numpy()], axis=0)
                    # dict_length = m_lens_0[0].item() + m_lens_1[0].item()
                    length_0 = pred_motions_0[0].shape[0]
                    length_1 = pred_motions_1[0].shape[0]
                    length = dict_motion.shape[0]


                    if t == 0:
                        sub_dict = {'motion_0': pred_motions_0[0].cpu().numpy(),
                                    'motion_1': pred_motions_1[0].cpu().numpy(),
                                    'length_0': length_0,
                                    'length_1': length_1,
                                    'cap_len_0': cap_lens_0[0].item(),
                                    'cap_len_1': cap_lens_1[0].item(),
                                    'caption_0': caption_0[0],
                                    'caption_1': caption_1[0],
                                    'tokens_0': tokens_0,
                                    'tokens_1': tokens_1,
                                    'motion': dict_motion,
                                    'length': length,
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens,
                                    'id_name': id_name
                                    }
                        generated_motion.append(sub_dict)


                    if is_mm:
                        mm_motions.append({
                            'motion_0': pred_motions_0[0].cpu().numpy(),
                            'motion_1': pred_motions_1[0].cpu().numpy(),
                            'length_0': length_0,
                            'length_1': length_1,
                            'motion': dict_motion,
                            'length': length
                        })
                if is_mm:
                    mm_generated_motions.append({'caption_0': caption_0[0],
                                                 'caption_1': caption_1[0],
                                                 'tokens_0': tokens_0,
                                                 'tokens_1': tokens_1,
                                                 'cap_len_0': cap_lens_0[0].item(),
                                                 'cap_len_1': cap_lens_1[0].item(),
                                                 'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

                    # if len(mm_generated_motions) < mm_num_samples:
                    #     print(len(mm_generated_motions), mm_idxs[len(mm_generated_motions)])
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        # print(len(generated_motion))
        # print(len(mm_generated_motions))
        self.opt = opt
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion_0, m_length_0, caption_0, tokens_0 = data['motion_0'], data['length_0'], data['caption_0'], data['tokens_0']
        motion_1, m_length_1, caption_1, tokens_1 = data['motion_1'], data['length_1'], data['caption_1'], data['tokens_1']
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len_0 = data['cap_len_0']
        sent_len_1 = data['cap_len_1']
        sent_len = data['cap_len']
        id_name = data['id_name']

        pos_one_hots_0 = []
        word_embeddings_0 = []
        for token in tokens_0:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots_0.append(pos_oh[None, :])
            word_embeddings_0.append(word_emb[None, :])
        pos_one_hots_0 = np.concatenate(pos_one_hots_0, axis=0)
        word_embeddings_0 = np.concatenate(word_embeddings_0, axis=0)

        pos_one_hots_1 = []
        word_embeddings_1 = []
        for token in tokens_1:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots_1.append(pos_oh[None, :])
            word_embeddings_1.append(word_emb[None, :])
        pos_one_hots_1 = np.concatenate(pos_one_hots_1, axis=0)
        word_embeddings_1 = np.concatenate(word_embeddings_1, axis=0)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length_0 < self.opt.max_motion_length:
            motion_0 = np.concatenate([motion_0,
                                       np.zeros((self.opt.max_motion_length - m_length_0, motion_0.shape[1]))
                                       ], axis=0)

        if m_length_1 < self.opt.max_motion_length:
            motion_1 = np.concatenate([motion_1,
                                       np.zeros((self.opt.max_motion_length - m_length_1, motion_1.shape[1]))
                                       ], axis=0)

        if m_length < self.opt.max_motion_length * 2:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length * 2 - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings_0, word_embeddings_1, pos_one_hots_0, pos_one_hots_1, caption_0, caption_1, \
               sent_len_0, sent_len_1, motion_0, motion_1, m_length_0, m_length_1, '_'.join(tokens_0), '_'.join(tokens_1), \
               word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name

class st2mV13GeneratedDatasetV2_reallen_slerp(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, tran_infer, his_text_enc = build_models_tran2_clip(opt)
        trainer = st2m_TrainerV13(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, tran_infer, his_text_enc,
                                 mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)


        min_mov_length = 4
        # print(mm_idxs)

        print('Loading model: Epoch %03d' % (epoch))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb_0, word_emb_1, pos_ohot_0, pos_ohot_1, caption_0, caption_1, \
                cap_lens_0, cap_lens_1, motions_0, motions_1, m_lens_0, m_lens_1, tokens_0, tokens_1, \
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens, id_name = data

                tokens_0 = tokens_0[0].split('_')
                tokens_1 = tokens_1[0].split('_')
                tokens = tokens[0].split('_')


                word_emb_0 = word_emb_0.detach().to(opt.device).float()
                word_emb_1 = word_emb_1.detach().to(opt.device).float()
                pos_ohot_0 = pos_ohot_0.detach().to(opt.device).float()
                pos_ohot_1 = pos_ohot_1.detach().to(opt.device).float()



                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                # if is_mm:
                #     print(mm_num_now, i, mm_idxs[mm_num_now])
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                # print(m_lens[0].item(), cap_lens[0].item())
                for t in range(repeat_times):
                    mov_len_0 = torch.div(m_lens_0[0], opt.unit_length, rounding_mode='trunc')
                    his_movements = trainer.mov_enc(
                        torch.zeros((word_emb_0.shape[0], opt.unit_length, opt.dim_pose - 4),
                                    device=opt.device)
                    ).repeat(1, mov_len_0, 1)  # (bs, mov_len, mov_dim)
                    his_real_mov_lens = torch.tensor(1).repeat(word_emb_0.shape[0]).to(opt.device)
                    his_caption = ['start'] * word_emb_0.shape[0]

                    pred_motions_0, _, _, his_movements, his_real_mov_lens = trainer.generate_transition_2code(
                        word_emb_0,
                        pos_ohot_0,
                        cap_lens_0,
                        m_lens_0,
                        mov_len_0,
                        opt.dim_pose,
                        his_movements,
                        his_real_mov_lens,
                        his_caption)

                    mov_len_1 = torch.div(m_lens_1[0], opt.unit_length, rounding_mode='trunc')
                    new_m_lens_1 = m_lens_1 - 4
                    new_mov_len_1 = mov_len_1 - 1
                    his_caption = [caption_0[0]]

                    pred_motions_1, _, _, his_movements, his_real_mov_lens = trainer.generate_transition_2code(
                        word_emb_1,
                        pos_ohot_1,
                        cap_lens_1,
                        new_m_lens_1,
                        new_mov_len_1,
                        opt.dim_pose,
                        his_movements,
                        his_real_mov_lens,
                        his_caption)

                    dict_motion = np.concatenate([pred_motions_0[0].cpu().numpy(), pred_motions_1[0].cpu().numpy()], axis=0)
                    length_0 = pred_motions_0[0].shape[0]
                    length_1 = pred_motions_1[0].shape[0]
                    length = dict_motion.shape[0]

                    slerp_gap_motion = torch.zeros([4, 263], dtype=torch.float64)
                    new_dict_motion = torch.cat((pred_motions_0[0].cpu(), slerp_gap_motion, pred_motions_1[0].cpu()), 0)
                    length_1 = length_1 + 4
                    length = new_dict_motion.shape[0]
                    new_slerp_lengths = [length_0, length_1]
                    new_dict_motion = do_slerp_op(new_dict_motion, new_slerp_lengths).numpy()
                    new_motions_1 = new_dict_motion[length_0:]



                    if t == 0:

                        sub_dict = {'motion_0': pred_motions_0[0].cpu().numpy(),
                                    'motion_1': new_motions_1,
                                    'length_0': length_0,
                                    'length_1': length_1,
                                    'cap_len_0': cap_lens_0[0].item(),
                                    'cap_len_1': cap_lens_1[0].item(),
                                    'caption_0': caption_0[0],
                                    'caption_1': caption_1[0],
                                    'tokens_0': tokens_0,
                                    'tokens_1': tokens_1,
                                    'motion': new_dict_motion,
                                    'length': length,
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens,
                                    'id_name': id_name
                                    }
                        generated_motion.append(sub_dict)


                    if is_mm:
                        mm_motions.append({
                            'motion_0': pred_motions_0[0].cpu().numpy(),
                            'motion_1': new_motions_1,
                            'length_0': length_0,
                            'length_1': length_1,
                            'motion': new_dict_motion,
                            'length': length
                        })
                if is_mm:
                    mm_generated_motions.append({'caption_0': caption_0[0],
                                                 'caption_1': caption_1[0],
                                                 'tokens_0': tokens_0,
                                                 'tokens_1': tokens_1,
                                                 'cap_len_0': cap_lens_0[0].item(),
                                                 'cap_len_1': cap_lens_1[0].item(),
                                                 'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

                    # if len(mm_generated_motions) < mm_num_samples:
                    #     print(len(mm_generated_motions), mm_idxs[len(mm_generated_motions)])
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        # print(len(generated_motion))
        # print(len(mm_generated_motions))
        self.opt = opt
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion_0, m_length_0, caption_0, tokens_0 = data['motion_0'], data['length_0'], data['caption_0'], data['tokens_0']
        motion_1, m_length_1, caption_1, tokens_1 = data['motion_1'], data['length_1'], data['caption_1'], data['tokens_1']
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len_0 = data['cap_len_0']
        sent_len_1 = data['cap_len_1']
        sent_len = data['cap_len']
        id_name = data['id_name']

        pos_one_hots_0 = []
        word_embeddings_0 = []
        for token in tokens_0:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots_0.append(pos_oh[None, :])
            word_embeddings_0.append(word_emb[None, :])
        pos_one_hots_0 = np.concatenate(pos_one_hots_0, axis=0)
        word_embeddings_0 = np.concatenate(word_embeddings_0, axis=0)

        pos_one_hots_1 = []
        word_embeddings_1 = []
        for token in tokens_1:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots_1.append(pos_oh[None, :])
            word_embeddings_1.append(word_emb[None, :])
        pos_one_hots_1 = np.concatenate(pos_one_hots_1, axis=0)
        word_embeddings_1 = np.concatenate(word_embeddings_1, axis=0)

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length_0 < self.opt.max_motion_length:
            motion_0 = np.concatenate([motion_0,
                                       np.zeros((self.opt.max_motion_length - m_length_0, motion_0.shape[1]))
                                       ], axis=0)

        if m_length_1 < self.opt.max_motion_length:
            motion_1 = np.concatenate([motion_1,
                                       np.zeros((self.opt.max_motion_length - m_length_1, motion_1.shape[1]))
                                       ], axis=0)

        if m_length < self.opt.max_motion_length * 2:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length * 2 - m_length, motion.shape[1]))
                                     ], axis=0)

        return word_embeddings_0, word_embeddings_1, pos_one_hots_0, pos_one_hots_1, caption_0, caption_1, \
               sent_len_0, sent_len_1, motion_0, motion_1, m_length_0, m_length_1, '_'.join(tokens_0), '_'.join(tokens_1), \
               word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), id_name