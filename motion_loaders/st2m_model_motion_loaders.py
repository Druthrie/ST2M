from torch.utils.data import DataLoader, Dataset
from utils.get_opt import get_opt

from motion_loaders.st2m_v13_model_dataset import st2mV13GeneratedDatasetV2_reallen, st2mV13GeneratedDatasetV2_reallen_slerp
from utils.word_vectorizer import WordVectorizer
import numpy as np
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)



class MMGeneratedDatasetV2(Dataset):
    def __init__(self, opt, motion_dataset, w_vectorizer):
        self.opt = opt
        self.dataset = motion_dataset.mm_generated_motion
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        # m_lens_0 = []
        # m_lens_1 = []
        m_lens = []
        # motions_0 = []
        # motions_1 = []
        motions = []
        for mm_motion in mm_motions:
            # m_lens_0.append(mm_motion['length_0'])
            # m_lens_1.append(mm_motion['length_1'])
            m_lens.append(mm_motion['length'])
            # motion_0 = mm_motion['motion_0']
            # motion_1 = mm_motion['motion_1']
            motion = mm_motion['motion']
            # if len(motion_0) < self.opt.max_motion_length:
            #     motion_0 = np.concatenate([motion_0,
            #                              np.zeros((self.opt.max_motion_length - len(motion_0), motion_0.shape[1]))
            #                              ], axis=0)
            # if len(motion_1) < self.opt.max_motion_length:
            #     motion_1 = np.concatenate([motion_1,
            #                              np.zeros((self.opt.max_motion_length - len(motion_1), motion_1.shape[1]))
            #                              ], axis=0)
            if len(motion) < self.opt.max_motion_length*2:
                motion = np.concatenate([motion,
                                         np.zeros((self.opt.max_motion_length*2 - len(motion), motion.shape[1]))
                                         ], axis=0)
            # motion_0 = motion_0[None, :]
            # motion_1 = motion_1[None, :]
            motion = motion[None, :]
            # motions_0.append(motion_0)
            # motions_1.append(motion_1)
            motions.append(motion)
        # m_lens_0 = np.array(m_lens_0, dtype=np.int)
        # m_lens_1 = np.array(m_lens_1, dtype=np.int)
        m_lens = np.array(m_lens, dtype=np.int)
        # motions_0 = np.concatenate(motions_0, axis=0)
        # motions_1 = np.concatenate(motions_1, axis=0)
        motions = np.concatenate(motions, axis=0)
        # sort_indx_0 = np.argsort(m_lens_0)[::-1].copy()
        # sort_indx_1 = np.argsort(m_lens_1)[::-1].copy()
        sort_indx = np.argsort(m_lens)[::-1].copy()

        # m_lens_0 = m_lens_0[sort_indx_0]
        # m_lens_1 = m_lens_1[sort_indx_1]
        m_lens = m_lens[sort_indx]
        # motions_0 = motions_0[sort_indx_0]
        # motions_1 = motions_1[sort_indx_1]
        motions = motions[sort_indx]
        # return motions_0, motions_1, m_lens_0, m_lens_1, motions, m_lens
        return motions, m_lens





def get_motion_loaderV2(opt_path, batch_size, ground_truth_dataset, mm_num_samples, mm_num_repeats, device, TEACH_repeat_now):
    opt = get_opt(opt_path, device)

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 'BABEL_TEACH':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    elif opt.dataset_name == 'STDM':
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    if opt.name == 'trainV13_LV1LT1LK001LA01_BABEL_TEACH' or opt.name == 'trainV13_LV1LT1LK001LA01_STDM':
        dataset = st2mV13GeneratedDatasetV2_reallen(opt, ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)
    elif opt.name == 'trainV13_LV1LT1LK001LA01_BABEL_TEACH_slerp' or opt.name == 'trainV13_LV1LT1LK001LA01_STDM_slerp':
        dataset = st2mV13GeneratedDatasetV2_reallen_slerp(opt, ground_truth_dataset, w_vectorizer, mm_num_samples, mm_num_repeats)
    else:
        raise KeyError('Dataset not recognized!!')

    mm_dataset = MMGeneratedDatasetV2(opt, dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader

