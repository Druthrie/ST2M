import os

from os.path import join as pjoin

import torch

import utils.paramUtil as paramUtil
from options.evaluate_options import st2m_TestOptions
from torch.utils.data import DataLoader
from utils.plot_script import *

from networks.modules import *
from networks.st2m_trainers import st2m_TrainerV13
from data.st2m_dataset import RawTextDataset
from scripts.motion_process import *
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.utils import *
from scripts.slerp import do_slerp_op

def plot_t2m(data, save_animation_dir, save_joints_dir, captions, dataset, is_all=False):
    data = dataset.inv_transform(data)
    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        # print(joint_data.shape)
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        # print(joint.shape)
        # save_path = '%s_%02d'%(save_dir, i)
        # np.save(save_path + '.npy', joint)
        # plot_3d_motion(save_path + '.mp4', paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)

        joint = motion_temporal_filter(joint, sigma=1)
        # print(joint.shape)
        np.save(save_joints_dir, joint)

        plot_3d_motion(save_animation_dir, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
        # if is_all:
        #     plot_3d_motion(save_animation_dir, paramUtil.t2m_kinematic_chain, joint, title='all', fps=20)
        # else:
        #     plot_3d_motion(save_animation_dir, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)


def loadDecompModel(opt):
    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_enc'])

    return movement_enc, movement_dec




def build_models_tran2_clip(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=dim_word,
                                        pos_size=dim_pos_ohot,
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

    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    tran_inferencer = TRANSITION_TRANSFORMER_2code(max_mov_len=opt.max_motion_length // opt.unit_length,
                                                   device=opt.device)

    his_text_encoder = TextEncoderCLIP(hidden_size=opt.dim_dec_hidden,
                                       device=opt.device)

    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, tran_inferencer, his_text_encoder


if __name__ == '__main__':
    parser = st2m_TestOptions()
    opt = parser.parse()
    opt.do_denoise = True

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    opt.result_dir = pjoin(opt.result_path, opt.dataset_name, opt.name, opt.ext)
    opt.joints_vecs_dir = pjoin(opt.result_dir, 'joints_vecs')
    opt.joints_dir = pjoin(opt.result_dir, 'joints')
    opt.animation_dir = pjoin(opt.result_dir, 'animations')
    os.makedirs(opt.joints_vecs_dir, exist_ok=True)
    os.makedirs(opt.joints_dir, exist_ok=True)
    os.makedirs(opt.animation_dir, exist_ok=True)


    if opt.dataset_name == 'BABEL_TEACH':
        opt.data_root = './dataset/BABEL_TEACH'
        opt.motion_dir = pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'BABEL_TEACH_texts')
        opt.joints_num = 22
        dim_pose = 263
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        num_classes = 200 // opt.unit_length

        fps = 20

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, opt.split_file)
        opt.max_motion_length = 196

    elif opt.dataset_name == 'STDM':
        opt.data_root = './dataset/STDM'
        opt.motion_dir = pjoin(opt.data_root, 'STDM_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'STDM_texts_5289')
        opt.joints_num = 22
        dim_pose = 263
        dim_word = 300
        dim_pos_ohot = len(POS_enumerator)
        num_classes = 200 // opt.unit_length

        fps = 20

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, opt.split_file)
        opt.max_motion_length = 196

    else:
        raise KeyError('Dataset Does Not Exist')


    text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, tran_infer, his_text_enc = build_models_tran2_clip(opt)

    trainer = st2m_TrainerV13(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, tran_infer, his_text_enc,
                             mov_enc=mov_enc)


    dataset = RawTextDataset(opt, mean, std, opt.text_file, w_vectorizer)

    epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
    print('Loading model: Epoch %03d'%(epoch))
    trainer.eval_mode()
    trainer.to(opt.device)

    data_loader = DataLoader(dataset, batch_size=1, drop_last=True, num_workers=1)

    for repeat_id in range(opt.repeat_times):
        print('------------- round %02d ---------------' % repeat_id)
        '''Generate Results'''
        print('Generate Results')
        is_slerp = opt.do_slerp
        new_slerp_lengths = []
        result_dict = {}
        time_count = 0
        time_avg = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                print('%02d_%03d' % (i, len(data_loader)))
                word_emb, pos_ohot, caption, cap_lens, dur = data

                name = 'C%03d' % (i)
                item_dict = {'caption': caption}
                print(caption)

                word_emb, pos_ohot, caption, cap_lens, dur = data
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                m_lens = dur.detach().to(opt.device) * fps
                # print(length.item())

                if m_lens[0].data > 196:
                    m_lens = torch.tensor([196, ], dtype=torch.int64).to(opt.device)
                print('m_lens: ', m_lens, m_lens.shape[0])
                max_mov_len = torch.tensor(49).to(opt.device)
                # print("debug")

                mov_lens = torch.div(m_lens[0], opt.unit_length, rounding_mode='trunc')
                if i == 0:
                    time1 = time.time()
                    his_movements = trainer.mov_enc(
                        torch.zeros((word_emb.shape[0], opt.unit_length, dim_pose - 4),
                                    device=opt.device)
                    ).repeat(1, mov_lens, 1)  # (bs, mov_len, mov_dim)
                    his_real_mov_lens = torch.tensor(1).repeat(word_emb.shape[0]).to(opt.device)
                    his_caption = ['start'] * word_emb.shape[0]
                    # print('his_caption:', his_caption)

                    pred_motions, _, att_wgts, his_movements, his_real_mov_lens = trainer.generate_transition_2code(
                        word_emb,
                        pos_ohot,
                        cap_lens,
                        m_lens,
                        mov_lens,
                        dim_pose,
                        his_movements,
                        his_real_mov_lens,
                        his_caption)
                    time2 = time.time()
                    time_count += 1
                    time_avg += time2 - time1
                    print('gen_time: %5f s' % ((time2 - time1)))

                    his_caption = [caption[0]]
                else:
                    print('his_caption:', his_caption)

                    time1 = time.time()
                    if is_slerp:
                        new_m_lens = m_lens - 4
                        new_mov_lens = mov_lens - 1
                        pred_motions, _, att_wgts, his_movements, his_real_mov_lens = trainer.generate_transition_2code(
                            word_emb,
                            pos_ohot,
                            cap_lens,
                            new_m_lens,
                            new_mov_lens,
                            dim_pose,
                            his_movements,
                            his_real_mov_lens,
                            his_caption)
                    else:
                        pred_motions, _, att_wgts, his_movements, his_real_mov_lens = trainer.generate_transition_2code(
                            word_emb,
                            pos_ohot,
                            cap_lens,
                            m_lens,
                            mov_lens,
                            dim_pose,
                            his_movements,
                            his_real_mov_lens,
                            his_caption)

                    time2 = time.time()
                    time_count += 1
                    time_avg += time2 - time1
                    print('gen_time: %5f s' % ((time2 - time1)))

                    his_caption = [caption[0]]
                if is_slerp:
                    if i == 0:
                        new_slerp_lengths.append(m_lens[0])
                    else:
                        slerp_gap_motion = torch.zeros([1, 4, 263], dtype=torch.float64)
                        pred_motions = torch.cat((slerp_gap_motion, pred_motions.cpu()), 1)
                        new_slerp_lengths.append(m_lens[0])

                sub_dict = {}
                sub_dict['motion'] = pred_motions.cpu().numpy()
                sub_dict['att_wgts'] = att_wgts.cpu().numpy()
                sub_dict['m_len'] = m_lens[0]
                item_dict['result'] = sub_dict
                result_dict[name] = item_dict

        time_avg = time_avg / time_count
        print('time_avg: ', time_avg)

        print('Animation Results')
        '''Animate Results'''

        list_motion = []
        list_caption = []
        for i, (key, item) in enumerate(result_dict.items()):
            print('%02d_%03d' % (i, len(result_dict)))
            captions = item['caption']
            sub_dict = item['result']
            motion = sub_dict['motion']
            print(motion.shape)
            # save every text result
            # if is_slerp == False:
            #     att_wgts = sub_dict['att_wgts']
            #     joints_vec = motion.copy()
            #     joints_vec.resize(motion.shape[1], motion.shape[2])
            #     save_joints_vec_dir = pjoin(opt.joints_vecs_dir, 'C%03d_joints_vec_L%03d_R%02d.npy' % (i, motion.shape[1], repeat_id))
            #     np.save(save_joints_vec_dir, joints_vec)
            #     save_animation_dir = pjoin(opt.animation_dir, 'C%03d_animation_L%03d_R%02d.mp4' % (i, motion.shape[1], repeat_id))
            #     save_joints_dir = pjoin(opt.joints_dir, 'C%03d_joints_L%03d_R%02d.npy' % (i, motion.shape[1], repeat_id))
            #     plot_t2m(motion, save_animation_dir, save_joints_dir, captions, dataset)
            if i == 0:
                sum_motion = motion
            else:
                sum_motion = np.concatenate([sum_motion, motion], axis=1)
            list_motion.append(motion)
            if i == 0:
                list_caption = captions[0]
            else:
                list_caption = list_caption + ' --> ' + captions[0]
        list_caption = (list_caption,)

        if is_slerp:
            new_dict_motion = torch.tensor(sum_motion[0])
            new_dict_motion = do_slerp_op(new_dict_motion, new_slerp_lengths).numpy()
            sum_motion = np.expand_dims(new_dict_motion, axis=0)

        save_all_joints_vec_dir = pjoin(opt.joints_vecs_dir, 'all_joints_vec_L%03d_R%02d.npy' % (sum_motion.shape[1], repeat_id))
        sum_joints_vec = sum_motion.copy()
        sum_joints_vec.resize(sum_motion.shape[1], sum_motion.shape[2])
        np.save(save_all_joints_vec_dir, sum_joints_vec)
        save_all_joints_dir = pjoin(opt.joints_dir, 'all_joints_L%03d_R%02d.npy' % (sum_motion.shape[1], repeat_id))
        save_all_animation_dir = pjoin(opt.animation_dir, 'all_animation_L%03d_R%02d.mp4' % (sum_motion.shape[1], repeat_id))

        plot_t2m(sum_motion, save_all_animation_dir, save_all_joints_dir, list_caption, dataset)



