import os

from os.path import join as pjoin
import torch
from options.train_options import st2m_TrainTexMotMatchOptions

from networks.modules import *
from networks.st2m_trainers import st2m_TextMotionMatchTrainer
from data.st2m_dataset import st2m_Text2Motion_withpast_Dataset_match, collate_fn
from scripts.motion_process import *
from torch.utils.data import DataLoader
from utils.word_vectorizer import WordVectorizer, POS_enumerator


def build_models(opt):
    movement_enc = MovementConvEncoder(dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    if not opt.is_continue:
       checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                               map_location=opt.device)
       movement_enc.load_state_dict(checkpoint['movement_enc'])
    return text_enc, motion_enc, movement_enc


if __name__ == '__main__':
    parser = st2m_TrainTexMotMatchOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    opt.eval_dir = pjoin(opt.save_root, 'eval')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == 'BABEL_TEACH':
        opt.data_root = './dataset/BABEL_TEACH'
        opt.motion_dir = pjoin(opt.data_root, 'BABEL_TEACH_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'BABEL_TEACH_texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        num_classes = 200 // opt.unit_length
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'trainV13_LV1LT1LK001LA01_BABEL_TEACH', 'meta')
        train_split_file = pjoin(opt.data_root, 'train_12103.txt')
        val_split_file = pjoin(opt.data_root, 'val_2163.txt')
    elif opt.dataset_name == 'STDM':
        opt.data_root = './dataset/STDM'
        opt.motion_dir = pjoin(opt.data_root, 'STDM_joint_vecs_2')
        opt.text_dir = pjoin(opt.data_root, 'STDM_texts_5289')
        opt.joints_num = 22
        opt.max_motion_length = 196
        dim_pose = 263
        num_classes = 200 // opt.unit_length
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'train13_LV1LT1LK001LA01_STDM', 'meta')
        train_split_file = pjoin(opt.data_root, 'train_4231.txt')
        val_split_file = pjoin(opt.data_root, 'val_265.txt')
    else:
        raise KeyError('Dataset Does Not Exist')

    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    text_encoder, motion_encoder, movement_encoder = build_models(opt)

    pc_text_enc = sum(param.numel() for param in text_encoder.parameters())
    print(text_encoder)
    print("Total parameters of text encoder: {}".format(pc_text_enc))
    pc_motion_enc = sum(param.numel() for param in motion_encoder.parameters())
    print(motion_encoder)
    print("Total parameters of motion encoder: {}".format(pc_motion_enc))
    print("Total parameters: {}".format(pc_motion_enc + pc_text_enc))


    trainer = st2m_TextMotionMatchTrainer(opt, text_encoder, motion_encoder, movement_encoder)

    train_dataset = st2m_Text2Motion_withpast_Dataset_match(opt, mean, std, train_split_file, w_vectorizer)
    val_dataset = st2m_Text2Motion_withpast_Dataset_match(opt, mean, std, val_split_file, w_vectorizer)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers,
                              shuffle=True, collate_fn=collate_fn, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=opt.num_workers,
                            shuffle=True, collate_fn=collate_fn, pin_memory=True)

    trainer.train(train_loader, val_loader)