from fedscale.core.aggregation.aggregator import Aggregator
import collections


class EvoFed_Aggregator(Aggregator):
    def __init__(self, args, conf):
        super().__init__(args)
        # ======== override fedscale attributes ========
        self.model = [None]
        self.model_in_update = [0]
        self.model_weights = [collections.OrderedDict()]
        self.last_gradient_weights = [[]]
        self.model_state_dict = [None]
        self.tasks_round = [0]
        self.model_update_size = [0.]

        # ======== evofed specific attributes ========
        self.layer_rankings = []
        self.global_training_loss = []
        self.local_training_loss = collections.defaultdict(list)
        self.global_training_time = []
        self.local_training_time = collections.defaultdict(list)
        self.server_config = conf

    def init_model(self):
        if self.server_config['model_source'] == 'fedscale':
            self.model = 


