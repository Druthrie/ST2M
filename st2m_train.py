import os

from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import st2m_Train_withpast_Options
from utils.plot_script import *

from networks.modules import *
from networks.st2m_trainers import st2m_TrainerV13
from data.st2m_dataset import st2m_Text2Motion_withpast_DatasetV4
from scripts.motion_process import *
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.utils import *

def plot_t2m(data, save_dir, captions, ep_curves=None):
    data = train_dataset.inv_transform(data)
    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)
        # print(ep_curve.shape)
        if ep_curves is not None:
            ep_curve = ep_curves[i]
            plt.plot(ep_curve)
            plt.title(caption)
            save_path = pjoin(save_dir, '%02d.png' % (i))
            plt.savefig(save_path)
            plt.close()

def stdm_plot_t2m_eval(data, save_dir, captions, real_mot_lens, mul_data_size, use_reallen=True):
    data = train_dataset.inv_transform(data)

    for group in range(3):
        if group == 0:
            data_type = 'fake_motion'
        elif group == 1:
            data_type = 'reco_motion'
        else:
            data_type = 'gt_motion'
        for i in range(mul_data_size):
            real_len = real_mot_lens[i % mul_data_size]
            joint_data = data[group*mul_data_size + i]
            if use_reallen:
                joint_data = joint_data[:real_len]
            np.save(pjoin(save_dir, data_type + '_joint_vecs_C%03d.npy' % (i)), joint_data)
            caption = captions[group*mul_data_size + i]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
            # joint = motion_temporal_filter(joint, sigma=1)
            np.save(pjoin(save_dir, data_type + '_joints_C%03d.npy' % (i)), joint)
            save_path = pjoin(save_dir, data_type + '_animation_C%03d.mp4' % (i))
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=fps, radius=radius)

            if i == 0:
                mul_joint_data = joint_data
                mul_caption = caption
            else:
                mul_joint_data = np.concatenate([mul_joint_data, joint_data], axis=0)
                mul_caption = mul_caption + ' --> ' + caption

            if i == mul_data_size-1:
                np.save(pjoin(save_dir, data_type + '_joint_vecs_all.npy'), mul_joint_data)
                mul_joint = recover_from_ric(torch.from_numpy(mul_joint_data).float(), opt.joints_num).numpy()
                # mul_joint = motion_temporal_filter(mul_joint, sigma=1)
                np.save(pjoin(save_dir, data_type + '_joints_all.npy'), mul_joint)
                save_path = pjoin(save_dir, data_type + '_animation_all.mp4')
                plot_3d_motion(save_path, kinematic_chain, mul_joint, title=mul_caption, fps=fps, radius=radius)






def loadDecompModel(opt):
    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, dim_pose)

    if not opt.is_continue:
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                                map_location=opt.device)
        movement_enc.load_state_dict(checkpoint['movement_enc'])
        movement_dec.load_state_dict(checkpoint['movement_dec'])

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

    seq_posterior = TextDecoder(text_size=text_size,
                                input_size=opt.dim_att_vec + opt.dim_movement_latent * 2,
                                output_size=opt.dim_z,
                                hidden_size=opt.dim_pos_hidden,
                                n_layers=opt.n_layers_pos)

    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    tran_inferencer = TRANSITION_TRANSFORMER_2code(max_mov_len=opt.max_motion_length // opt.unit_length,
                                                   device=opt.device)

    his_text_encoder = TextEncoderCLIP(hidden_size=opt.dim_dec_hidden,
                                       device=opt.device)

    return text_encoder, seq_prior, seq_posterior, seq_decoder, att_layer, tran_inferencer, his_text_encoder


if __name__ == '__main__':
    parser = st2m_Train_withpast_Options()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 'BABEL_TEACH':
        opt.data_root = './dataset/BABEL_TEACH'
        opt.motion_dir = pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'BABEL_TEACH_texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
        train_split_file = pjoin(opt.data_root, 'train_12103.txt')
        val_split_file = pjoin(opt.data_root, 'val_2163.txt')
    elif opt.dataset_name == 'STDM':
        opt.data_root = './dataset/STDM'
        opt.motion_dir = pjoin(opt.data_root, 'STDM_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'STDM_texts_5289')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
        train_split_file = pjoin(opt.data_root, 'train_4231.txt')
        val_split_file = pjoin(opt.data_root, 'val_265.txt')
    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    movement_enc, movement_dec = loadDecompModel(opt)
    text_encoder, seq_prior, seq_posterior, seq_decoder, att_layer, tran_infer, his_text_encoder = build_models_tran2_clip(opt)

    print(text_encoder)
    print(seq_prior)
    print(seq_posterior)
    print(seq_decoder)
    print(att_layer)
    print(tran_infer)
    print(his_text_encoder)
    print([name for name, p in his_text_encoder.named_parameters() if p.requires_grad])

    trainer = st2m_TrainerV13(opt, text_encoder, seq_prior, seq_decoder, att_layer, movement_dec,
                             tran_infer, his_text_encoder, mov_enc=movement_enc, seq_post=seq_posterior)

    train_dataset = st2m_Text2Motion_withpast_DatasetV4(opt, mean, std, train_split_file, w_vectorizer)
    val_dataset = st2m_Text2Motion_withpast_DatasetV4(opt, mean, std, val_split_file, w_vectorizer)

    trainer.train(train_dataset, val_dataset, plot_t2m, stdm_plot_t2m_eval)

