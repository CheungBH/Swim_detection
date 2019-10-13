from utils.utils import Utils


class CosineTracker(object):
    def __init__(self, pre_info, curr_info):
        self.pre_info = pre_info
        self.curr_info = curr_info
        self.cos_ls = []
        self.cal_cos = Utils.cal_cosine

    def get_matrix(self):
        





