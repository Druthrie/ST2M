from options.base_options import BaseOptions
import argparse

class st2m_TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--start_mov_len', type=int, default=10)
        # self.parser.add_argument('--est_length', action="store_true", help="Whether to use sampled motion length")


        self.parser.add_argument('--repeat_times', type=int, default=3, help="Number of generation rounds for each text description")
        self.parser.add_argument('--split_file', type=str, default='test_2000.txt')
        self.parser.add_argument('--text_file', type=str, default="./inputs_texts/teach_16-196/0.txt",
                                 help='Path of text description for motion generation')
        self.parser.add_argument('--which_epoch', type=str, default="latest", help='Checkpoint that will be used')
        self.parser.add_argument('--result_path', type=str, default="./eval_results/", help='Path to save generation results')
        # self.parser.add_argument('--num_results', type=int, default=2, help='Number of descriptions that will be used')
        self.parser.add_argument('--ext', type=str, default='ext', help='Save file path extension')
        self.parser.add_argument('--do_slerp', action="store_true", help='Save file path extension')

        self.is_train = False

class st2m_finalEvalOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--dataset_name', type=str, default='STDM', help='Dataset Name')
        self.parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
        self.parser.add_argument('--mm_num_samples', type=int, default=100)
        self.parser.add_argument('--mm_num_repeats', type=int, default=30)
        self.parser.add_argument('--mm_num_times', type=int, default=10)
        self.parser.add_argument('--diversity_times', type=int, default=300)
        self.parser.add_argument('--replication_times', type=int, default=20)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--log_file_name', type=str, default='final_contrast')


    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        return self.opt