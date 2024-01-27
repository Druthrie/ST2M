import torch

def slerp_translation(last_transl, new_transl, number_of_frames):
    alpha = torch.linspace(0, 1, number_of_frames+2)
    # 2 more than needed
    inter_trans = torch.einsum("i,...->i...", 1-alpha, last_transl) + torch.einsum("i,...->i...", alpha, new_transl)
    return inter_trans[1:-1]

def do_slerp_op(motion, lengths, slerp_window_size=4):
    pose = motion.clone()
    end_first_motion = lengths[0] - 1
    for length in lengths[1:]:
        begin_second_motion = end_first_motion + 1
        begin_second_motion += slerp_window_size

        inter_pose = slerp_translation(pose[end_first_motion], pose[begin_second_motion], slerp_window_size)
        pose[end_first_motion + 1:begin_second_motion] = inter_pose

        end_first_motion += length
    return pose