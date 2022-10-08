from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core.fllibs import parser
import fedscale.core.logger.aggragation as logger
import fedscale.core.commons as commons
import collections
import pickle
import sys
import copy
import logging
import torch
import math
import yaml
import os
import time
import numpy as np
from model_manager import Model_Manager
from evofed.model.init_model import init_model
from model_manager import shape_match, get_flops
from client_manager import Evofed_ClientManager


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
        self.loss_accumulator = collections.defaultdict(list)
        self.client_manager = self.init_client_manager(self.args)

        # ======== evofed specific attributes ========
        self.model_weights_name = []
        # weight_name = self.model_weights_name[model_id][weight_id]
        self.model_weights_weights = []
        # weight_weight = self.model_weights_weights[model_id][weight_name]
        self.model_flops = []
        self.latest_model_layer_rankings = []
        self.layer_gradients = [collections.defaultdict(list)]
        # layer_gradient = self.layer_gradients[model_id][layer_name][round]
        self.global_training_loss = collections.defaultdict(list)
        self.local_training_loss = collections.defaultdict(list)
        self.global_training_time = []
        self.local_training_time = collections.defaultdict(list)
        self.server_config = conf
        self.flatten_client_duration = []
        self.saved_id = 0

    def init_client_manager(self, args):
        client_manager = Evofed_ClientManager(args.sample_mode, args=args)
        return client_manager

    def init_model(self):
        # check model source, enable multiple model init
        if self.server_config['model_source'] == 'fedscale':
            models = [init_model(
                self.args.task, self.args.model, self.args.data_set)]
        elif self.server_config['model_source'] == 'load':
            models = []
            model_paths = self.server_config['model_path'].split(
                ',') if ',' in self.server_config['model_path'] else [self.server_config['model_path']]
            for model_path in model_paths:
                model_path = model_path.strip()
                with open(f"FedScale/checkpoints/{model_path}.pth.tar", 'rb') as f:
                    model = pickle.load(f)
                    models.append(model)
            logging.info(
                f"loading {len(models)} models from server: {model_paths}")
        else:
            models = [init_model(
                self.args.task, self.server_config['model_source'], self.args.data_set)]

        self.model = models
        self.model_weights = [model.state_dict() for model in self.model]
        for model_weight in self.model_weights:
            self.model_weights_name.append(
                [param_name for param_name in model_weight])
            self.model_weights_weights.append(collections.defaultdict(list))
        self.model_manager = Model_Manager(int(self.server_config['seed']))
        self.last_gradient_weights = [[] for _ in self.model]
        self.model_manager.load_models(models)
        self.layer_gradients = [
            collections.defaultdict(list) for _ in self.model]
        for model in self.model:
            flops = get_flops(model)
            self.model_flops.append(flops)

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        # NOTE: We try to remove dummy events as much as possible in simulations,
        # by removing the stragglers/offline clients in overcommitment"""
        sampledClientsReal = []
        completionTimes = []
        completed_client_clock = {}
        # 1. remove dummy clients that are not available to the end of training
        for client_to_run in sampled_clients:
            client_cfg = self.client_conf.get(client_to_run, self.args)

            exe_cost = self.client_manager.getCompletionTime(client_to_run,
                                                             batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
                                                             upload_size=self.model_update_size[0], download_size=self.model_update_size[0])

            roundDuration = exe_cost['computation'] + \
                exe_cost['communication']
            # if the client is not active by the time of collection, we consider it is lost in this round
            if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                sampledClientsReal.append(client_to_run)
                completionTimes.append([client_to_run, roundDuration])
                completed_client_clock[client_to_run] = exe_cost

        num_clients_to_collect = min(
            num_clients_to_collect, len(completionTimes))
        # 2. get the top-k completions to remove stragglers
        sortedWorkersByCompletion = sorted(
            range(len(completionTimes)), key=lambda k: completionTimes[k][1])
        top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
        clients_to_run = [sampledClientsReal[k] for k in top_k_index]

        dummy_clients = [sampledClientsReal[k]
                         for k in sortedWorkersByCompletion[num_clients_to_collect:]]
        round_duration = completionTimes[top_k_index[-1]][1]
        completionTimes.sort(key=lambda l: l[1])

        return (clients_to_run, dummy_clients,
                completed_client_clock, round_duration,
                completionTimes[:num_clients_to_collect])

    def run(self):
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.save_last_param()
        self.model_update_size = [sys.getsizeof(
            pickle.dumps(model))/1024.0*8. for model in self.model]  # kbits
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file)

        self.event_monitor()

    def client_completion_handler(self, results):
        # disable aggregation optimization
        # ======== override fedscale ========
        client_id = results['clientId']
        model_id = results['model_id']

        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator[model_id].append(results['moving_loss'])
        self.local_training_loss[client_id].append(
            [self.round, model_id, results['moving_loss']])

        self.client_manager.register_feedback(results['clientId'], results['utility'],
                                              auxi=math.sqrt(
                                              results['moving_loss']),
                                              time_stamp=self.round,
                                              duration=self.virtual_client_clock[results['clientId']]['computation'] +
                                              self.virtual_client_clock[results['clientId']
                                                                        ]['communication']
                                              )

        # ======== aggregate weights ========
        self.update_lock.acquire()
        self.model_in_update[model_id] += 1
        # aggregate gradient
        for layer in results['gradient']:
            if self.model_in_update[model_id] == 1:
                self.layer_gradients[model_id][layer].append(
                    results['gradient'][layer])
            else:
                self.layer_gradients[model_id][layer][-1] += results['gradient'][layer]
        if self.model_in_update[model_id] == self.tasks_round[model_id]:
            for layer in self.layer_gradients[model_id]:
                self.layer_gradients[model_id][layer][-1] /= self.tasks_round[model_id]

        if self.server_config['aggregate_mode'] == 'normal':
            self.aggregate_client_weights(results)
        elif self.server_config['aggregate_mode'] == 'soft':
            self.soft_aggregate_client_weights(results)

        self.update_lock.release()

    def aggregate_client_weights(self, results):
        """May aggregate client updates on the fly

        Args:
            results (dictionary): client's training result

        [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
        H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        # importance = 1./self.tasks_round

        model_id = results['model_id']

        for p in results['update_weight']:
            param_weight = results['update_weight'][p]
            if isinstance(param_weight, list):
                param_weight = np.asarray(param_weight, dtype=np.float32)
            param_weight = torch.from_numpy(
                param_weight).to(device=self.device)

            if self.model_in_update[model_id] == 1:
                self.model_weights[model_id][p].data = param_weight
            else:
                self.model_weights[model_id][p].data += param_weight

        if self.model_in_update[model_id] == self.tasks_round[model_id]:
            for p in self.model_weights[model_id]:
                d_type = self.model_weights[model_id][p].data.dtype

                self.model_weights[model_id][p].data = (
                    self.model_weights[model_id][p]/float(self.tasks_round[model_id])).to(dtype=d_type)

    def get_soft_agg_weight(self, trained_model_id, aggregated_model_id):
        if trained_model_id == aggregated_model_id:
            return 1.0
        elif int(trained_model_id) > int(aggregated_model_id):
            logging.info(f"disable aggregating higher model to lower model")
            return 0.0
        else:
            return 0.1 / max(self.round - 100 * int(aggregated_model_id), 1)
        # return 1.0

    def soft_aggregate_client_weights(self, results):
        trained_model_id = results['model_id']
        for model_id in range(len(self.model)):
            for p in results['update_weight']:
                param_weight = results['update_weight'][p]
                if isinstance(param_weight, list):
                    param_weight = np.asarray(param_weight, dtype=np.float32)
                param_weight = torch.from_numpy(
                    param_weight
                ).to(device=self.device)
                if p in self.model_weights_name[model_id]:
                    d_type = self.model_weights[model_id][p].data.dtype
                    weight = self.get_soft_agg_weight(
                        trained_model_id, model_id)
                    if d_type == torch.int64 and trained_model_id != model_id:
                        continue
                    self.model_weights_weights[model_id][p].append(weight)
                    transformed_weight = shape_match(
                        param_weight, self.model_weights[model_id][p].data)
                    if sum(self.model_in_update) == 1:
                        self.model_weights[model_id][p].data = (
                            weight * transformed_weight).to(dtype=d_type)
                    else:
                        self.model_weights[model_id][p].data += (
                            weight * transformed_weight).to(dtype=d_type)

            if sum(self.model_in_update) == sum(self.tasks_round):
                # logging.info(f"soft aggregation history of model {model_id}: {self.model_weights_weights[model_id]}")
                for p in self.model_weights[model_id]:
                    d_type = self.model_weights[model_id][p].data.dtype
                    self.model_weights[model_id][p].data = (
                        self.model_weights[model_id][p] / float(
                            sum(self.model_weights_weights[model_id][p])
                        )
                    ).to(dtype=d_type)
                    self.model_weights_weights[model_id][p] = []

    def save_last_param(self):
        for model_id, model in enumerate(self.model):
            self.last_gradient_weights[model_id] = [
                p.data.clone() for p in self.model[model_id].parameters()]
            self.model_weights[model_id] = copy.deepcopy(
                self.model[model_id].state_dict())

    def round_weight_handler(self):
        if self.round > 1:
            for model_id, _ in enumerate(self.model):
                self.model[model_id].load_state_dict(
                    self.model_weights[model_id])

    def model_assign(self, clientsToRun):
        if self.server_config['model_assignment'] == 'naive':
            # naive assignment
            assignment = {}
            num_models = len(self.model)
            partition = len(clientsToRun) // num_models
            tasks = []
            for i, client in enumerate(clientsToRun):
                assignment[str(client)] = min(i // partition, num_models-1)
                if i % partition == 0 and len(tasks) < num_models:
                    tasks.append(1)
                else:
                    tasks[-1] += 1
        else:
            # elif self.server_config['model_assignment'] == 'round-robin':
            chosen_model_id = self.round % len(self.model)
            assignment = {}
            for client in clientsToRun:
                assignment[str(client)] = chosen_model_id
            tasks = []
            for model_id in range(len(self.model)):
                if model_id == chosen_model_id:
                    tasks.append(len(clientsToRun))
                else:
                    tasks.append(0)
        return assignment, tasks

    def transform_criterion(self) -> bool:
        if self.server_config['transform_criterion'] == 'converge':
            loss = self.global_training_loss[len(self.model)-1]
            M = int(self.server_config['converge_M'])
            N = int(self.server_config['converge_N'])
            if len(loss) >= M + N:
                slope_avg = .0
                for i in range(N):
                    slope_avg += abs(loss[-1-i] - loss[-1-i-M]) / M
                slope_avg = slope_avg / N
                if slope_avg < float(self.server_config['converge_C']):
                    logging.info(
                        f"average slope {slope_avg} < {float(self.server_config['converge_C'])}, ready to transform")
                    return True
                else:
                    logging.info(
                        f"average slope {slope_avg} >= {float(self.server_config['converge_C'])}, cannot transform")
                    return False
            else:
                return False
        else:
            return False

    def finish_training_criterion(self) -> bool:
        if len(self.model) == 1:
            return False
        loss = self.global_training_loss[0]
        M = int(self.server_config['converge_M'])
        N = int(self.server_config['converge_N'])
        if len(loss) >= M + N:
            slope_avg = .0
            for i in range(N):
                slope_avg += abs(loss[-1-i] - loss[-1-i-M]) / M
            slope_avg = slope_avg / N
            if slope_avg < float(self.server_config['drop_C']):
                logging.info(
                    f"average slope {slope_avg} < {float(self.server_config['drop_C'])}, ready to save and finish")
                return True
            else:
                logging.info(
                    f"average slope {slope_avg} >= {float(self.server_config['drop_C'])}, keep training")
                return False
        else:
            return False

    def select_layers(self) -> list:
        most_active_layer = self.latest_model_layer_rankings[-1][-1]
        max_gradient = self.layer_gradients[-1][most_active_layer][-1]
        gradient_threshold = max_gradient * \
            float(self.server_config['layer_selection_alpha'])
        active_layer = []
        for layer in self.layer_gradients[-1]:
            if self.layer_gradients[-1][layer][-1] >= gradient_threshold:
                active_layer.append(layer)
        return active_layer

    def transform_latest_model(self):
        selected_layers = self.select_layers()
        logging.info("FL Transform...")
        logging.info(f'selected layers {selected_layers} to transform')
        self.model_manager.efficient_model_scale(selected_layers)
        self.model.append(self.model_manager.model[-1])
        self.model_weights.append(self.model_manager.model[-1].state_dict())
        self.model_weights_name.append(
            [param_name for param_name in self.model_weights[-1]])
        self.model_weights_weights.append(collections.defaultdict(list))
        self.latest_model_layer_rankings = []
        self.last_gradient_weights.append([])
        self.layer_gradients.append(collections.defaultdict(list))
        self.model_flops.append(get_flops(self.model[-1]))

    def drop_oldest_model(self):
        logging.info("saving trained model 0")
        model_path = os.path.join(
            logger.logDir, 'model_'+str(self.saved_id)+'_trained.pth.tar')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model[self.saved_id], f)
        self.saved_id += 1
        self.model.pop(0)
        self.model_weights.pop(0)
        self.model_weights_name.pop(0)
        self.model_weights_weights.pop(0)
        self.last_gradient_weights.pop(0)
        self.layer_gradients.pop(0)
        self.model_flops.pop(0)
        self.model_manager.drop_parent_model()
        for model_id in self.global_training_loss:
            if model_id != 0:
                self.global_training_loss[model_id -
                                          1] = self.global_training_loss[model_id]
        self.global_training_loss.pop(len(self.global_training_loss) - 1)

    def round_gradient_handler(self):
        flattened_gradient = []
        for layer in self.layer_gradients[-1]:
            flattened_gradient.append(
                [layer, self.layer_gradients[-1][layer][-1]])
        flattened_gradient = sorted(flattened_gradient, key=lambda l: l[1])
        self.latest_model_layer_rankings.append(
            [l[0] for l in flattened_gradient])
        logging.info(f'current layer ranking: {flattened_gradient}')

    def get_round_duration(self, clientsToRun):
        round_duration = []
        for client in clientsToRun:
            client_cfg = self.client_conf.get(client, self.args)
            flops = self.model_flops[self.model_assignment[str(client)]]
            exe_cost = self.client_manager.getCompletionTime(client,
                                                             batch_size=client_cfg.batch_size, upload_step=client_cfg.local_steps,
                                                             upload_size=self.model_update_size[
                                                                 0], download_size=self.model_update_size[0],
                                                             model_flops=flops)
            round_duration.append(exe_cost['computation'] +
                                  exe_cost['communication'])
        return max(round_duration)

    def round_completion_handler(self):
        self.global_virtual_clock += self.round_duration
        self.round += 1

        self.global_training_time.append(self.round_duration)

        for client_id, training_time in self.flatten_client_duration:
            self.local_training_time[client_id].append(training_time)

        self.round_weight_handler()

        self.round_gradient_handler()

        avgUtilLastround = sum(self.stats_util_accumulator) / \
            max(1, len(self.stats_util_accumulator))

        for clientId in self.round_stragglers:
            self.client_manager.register_feedback(clientId, avgUtilLastround,
                                                  time_stamp=self.round,
                                                  duration=self.virtual_client_clock[clientId]['computation'] +
                                                  self.virtual_client_clock[clientId]['communication'],
                                                  success=False)

        avg_loss = {}
        for model_id in self.loss_accumulator:
            avg_loss[model_id] = sum(self.loss_accumulator[model_id]) / \
                max(1, len(self.loss_accumulator[model_id]))
            self.global_training_loss[model_id].append(avg_loss[model_id])

        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: " +
                     f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # decide if need to transform
        if self.transform_criterion():
            self.transform_latest_model()

        # decide if need to drop the oldest
        if self.finish_training_criterion():
            self.drop_oldest_model()

        # disable tensorboard
        # if len(self.loss_accumulator):
        #     self.log_train_result(avg_loss)

        # update select participants
        self.sampled_participants = self.select_participants(
            select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
        (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
            self.sampled_participants, self.args.num_participants)

        logging.info(f"Selected participants to run: {clientsToRun}")

        self.model_assignment, self.tasks_round = self.model_assign(
            clientsToRun)

        # update round duration according to flops
        round_duration = self.get_round_duration(clientsToRun)

        # issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)

        # update executors
        self.sampled_executors = list(self.individual_client_events.keys())

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = np.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = [0 for _ in self.model]
        self.test_result_accumulator = collections.defaultdict(list)
        self.stats_util_accumulator = []
        self.loss_accumulator = collections.defaultdict(list)
        self.update_default_task_config()

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def save_models(self):
        for i, model in enumerate(self.model):
            model_path = os.path.join(
                logger.logDir, 'model_'+str(i)+'.pth.tar')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

    def testing_completion_handler(self, client_id, results):
        # results['results'] is a dict {model_id: results}
        results = results['results']

        for model_id in results:
            self.test_result_accumulator[model_id].append(results[model_id])

        if len(self.test_result_accumulator[0]) == len(self.executors):
            self.aggregate_test_result()
            self.broadcast_events_queue.append(commons.START_ROUND)

            self.save_models()

    def aggregate_test_result(self):
        for model_id in self.test_result_accumulator:
            accumulator = self.test_result_accumulator[model_id][0]
            for i in range(1, len(self.test_result_accumulator[model_id])):
                if self.args.task == 'detection':
                    for key in accumulator:
                        if key == 'boxes':
                            for j in range(596):
                                accumulator[key][j] = accumulator[key][j] + \
                                    self.test_result_accumulator[model_id][i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[model_id][i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[model_id][i][key]
            if self.args.task == 'detection':
                top_1 = round(
                    accumulator['top_1']*100.0/len(self.test_result_accumulator[model_id]), 4)
                top_5 = round(
                    accumulator['top_5']*100.0/len(self.test_result_accumulator[model_id]), 4)
                loss = accumulator['test_loss']
            else:
                top_1 = round(
                    accumulator['top_1']/accumulator['test_len']*100.0, 4)
                top_5 = round(
                    accumulator['top_5']/accumulator['test_len']*100.0, 4)
                loss = accumulator['test_loss'] / accumulator['test_len']
            test_len = accumulator['test_len']
            logging.info("FL Testing of model {} in round {}, virtual_clock {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                         .format(model_id, self.round, self.global_virtual_clock, top_1, top_5, loss, test_len))

    def get_client_conf(self, clientId):
        conf = {
            'learning_rate': self.args.learning_rate,
            'model_id': self.model_assignment[str(clientId)],
            'layer_names': self.model_manager.get_layers(self.model_assignment[str(clientId)])
        }
        return conf

    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()

                if current_event in (commons.UPDATE_MODEL, commons.MODEL_TEST):
                    self.dispatch_client_events(current_event)

                elif current_event == commons.START_ROUND:

                    self.dispatch_client_events(commons.CLIENT_TRAIN)

                elif current_event == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.sever_events_queue) > 0:
                client_id, current_event, meta, data = self.sever_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(
                        self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == sum(self.tasks_round):
                        self.round_completion_handler()
                    # else:
                    #     logging.info(f'training progress: {len(self.stats_util_accumulator)}/{sum(self.tasks_round)}')

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data))

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)


if __name__ == "__main__":
    with open('FedScale/evofed/configs/server.yml') as f:
        config = yaml.safe_load(f)
    aggregator = EvoFed_Aggregator(parser.args, config)
    aggregator.run()
