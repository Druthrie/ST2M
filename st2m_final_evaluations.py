from datetime import datetime
import numpy as np
import torch
from motion_loaders.st2m_dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.st2m_model_motion_loaders import get_motion_loaderV2
from utils.get_opt import get_opt
from utils.metrics import *
from networks.st2m_evaluator_wrapper import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from scripts.motion_process import *
from utils import paramUtil
from utils.utils import *
from options.evaluate_options import st2m_finalEvalOptions

from os.path import join as pjoin

TEACH_repeat_now = 0

def plot_t2m(data, save_dir, captions):
    data = gt_dataset.inv_transform(data)
    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), wrapper_opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%(i))
        plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
        # print(ep_curve.shape)

torch.multiprocessing.set_sharing_strategy('file_system')



def evaluate_matching_scoreV2(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                _, _, _, _, _, _, _, _, _, _, _, _, _, _, \
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                       motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()


                argsmax = np.argsort(dist_mat, axis=1)

                top_k_mat = calculate_top_k(argsmax, top_k=3)

                top_k_count += top_k_mat.sum(axis=0)


                all_size += text_embeddings.shape[0]


                all_motion_embeddings.append(motion_embeddings.cpu().numpy())


            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict




def evaluate_fidV2(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            _, _, _, _, _, _, _, _, _, _, _, _, _, _, \
            _, _, _, sent_lens, motions, m_lens, _, _ = batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict




def evaluate_multimodalityV2(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])

                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))

        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict

def evaluate_transition(motion_loaders, file):
    tran_score_dict = OrderedDict({})
    print('========== Evaluating Transition Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        transition_score_sum = 0
        all_size = 0
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                _, _, _, _, _, _, _, _, _, _, m_length_0, m_length_1, _, _, _, _, _, _, motion, _, _, _ = batch

                for i in range(motion.shape[0]):
                    transition_temp = abs(motion[i][m_length_0[i] - 1] - motion[i][m_length_0[i]]).cpu().numpy()
                    transition_score_sum += np.mean(transition_temp)
                all_size += motion.shape[0]
            transition_score = transition_score_sum / all_size
            tran_score_dict[motion_loader_name] = transition_score

        print(f'---> [{motion_loader_name}] Transition Score: {transition_score:.4f}')
        print(f'---> [{motion_loader_name}] Transition Score: {transition_score:.4f}', file=file, flush=True)
    return tran_score_dict







def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval





def evaluationV2(log_file):
    global TEACH_repeat_now
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Transition': OrderedDict({})})
        for replication in range(replication_times):
            TEACH_repeat_now = replication
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_scoreV2(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fidV2(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodalityV2(mm_motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            tran_score_dict = evaluate_transition(motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]

            for key, item in tran_score_dict.items():
                if key not in all_metrics['Transition']:
                    all_metrics['Transition'][key] = [item]
                else:
                    all_metrics['Transition'][key] += [item]



        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)






if __name__ == '__main__':
    parser = st2m_finalEvalOptions()
    opt = parser.parse()
    if opt.dataset_name == 'BABEL_TEACH':
        dataset_opt_path = './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/opt.txt'
        eval_motion_loaders = {

            ################
            ## BABEL_TEACH Dataset##
            ################
            'trainV13_LV1LT1LK001LA01_BABEL_TEACH': lambda: get_motion_loaderV2(
                './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH/opt.txt',
                batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device, TEACH_repeat_now
            ),
            'trainV13_LV1LT1LK001LA01_BABEL_TEACH_slerp': lambda: get_motion_loaderV2(
                './checkpoints/BABEL_TEACH/trainV13_LV1LT1LK001LA01_BABEL_TEACH_slerp/opt.txt',
                batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device, TEACH_repeat_now
            )
        }
    elif opt.dataset_name == 'STDM':
        dataset_opt_path = './checkpoints/STDM/trainV13_LV1LT1LK001LA01_STDM/opt.txt'
        eval_motion_loaders = {

            ################
            ## STDM Dataset##
            ################

            'train13_LV1LT1LK001LA01_STDM': lambda: get_motion_loaderV2(
                './checkpoints/STDM/trainV13_LV1LT1LK001LA01_STDM/opt.txt',
                batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device, TEACH_repeat_now
            ),
            'train13_LV1LT1LK001LA01_STDM_slerp': lambda: get_motion_loaderV2(
                './checkpoints/STDM/trainV13_LV1LT1LK001LA01_STDM_slerp/opt.txt',
                batch_size, gt_dataset, mm_num_samples, mm_num_repeats, device, TEACH_repeat_now
            ),
        }
    else:
        raise KeyError('Dataset Does Not Exist')

    device_id = opt.gpu_id
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    mm_num_samples = opt.mm_num_samples
    mm_num_repeats = opt.mm_num_repeats
    mm_num_times = opt.mm_num_times

    diversity_times = opt.diversity_times
    replication_times = opt.replication_times
    batch_size = opt.batch_size

    gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    os.makedirs(pjoin('./eval_log', opt.dataset_name), exist_ok=True)
    log_file = pjoin('./eval_log', opt.dataset_name, opt.log_file_name + '.log')
    evaluationV2(log_file)

