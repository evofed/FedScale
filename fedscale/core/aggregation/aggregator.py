
        (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
            self.sampled_participants, self.args.num_participants)

        logging.info(f"Selected participants to run: {clientsToRun}")
        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.tasks_round = [0 for _ in range(0, len(self.probs))]
        for i in range(0, len(clientsToRun)):
            self.tasks_round[self.mapped_models[clientsToRun[i]]] += 1

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(
                self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants]

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = numpy.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = [0 for _ in range(0, len(self.probs))]
        self.test_result_accumulator = [[] for i in range(0, len(self.probs))]
        self.stats_util_accumulator = []
        self.client_training_results = []

        if self.round >= self.args.rounds: 
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def log_train_result(self, avg_loss):
        """Result will be post on TensorBoard"""
        self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.round)
        self.log_writer.add_scalar(
            'FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock/60.)
        self.log_writer.add_scalar(
            'FAR/round_duration (min)', self.round_duration/60., self.round)
        self.log_writer.add_histogram(
            'FAR/client_duration (min)', self.flatten_client_duration, self.round)

    def log_test_result(self):
        self.log_writer.add_scalar(
            'Test/round_to_loss', self.testing_history['perf'][self.round]['loss'], self.round)
        self.log_writer.add_scalar(
            'Test/round_to_accuracy', self.testing_history['perf'][self.round]['top_1'], self.round)
        self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.round]['loss'],
                                   self.global_virtual_clock/60.)
        self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.round]['top_1'],
                                   self.global_virtual_clock/60.)

    def deserialize_response(self, responses):
        return pickle.loads(responses)

    def serialize_response(self, responses):
        return pickle.dumps(responses)

    def testing_completion_handler(self, results):
        """Each executor will handle a subset of testing dataset"""

        results = results['results']

        # List append is thread-safe
        self.test_result_accumulator[self.test_model_id].append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator[self.test_model_id]) == len(self.executors):
            accumulator = self.test_result_accumulator[self.test_model_id][0]
            for i in range(1, len(self.test_result_accumulator[self.test_model_id])):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + \
                                    self.test_result_accumulator[self.test_model_id][i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[self.test_model_id][i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[self.test_model_id][i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
                                                            'model_id': self.test_model_id,
                                                            'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator[self.test_model_id]), 4),
                                                            'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator[self.test_model_id]), 4),
                                                            'loss': accumulator['test_loss'],
                                                            'test_len': accumulator['test_len']
                                                            }
            else:
                self.testing_history['perf'][self.round] = {'round': self.round, 'clock': self.global_virtual_clock,
                                                            'model_id': self.test_model_id,
                                                            'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                                                            'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                                                            'loss': accumulator['test_loss']/accumulator['test_len'],
                                                            'test_len': accumulator['test_len']
                                                            }

            logging.info("FL Testing in round: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                         .format(self.round, self.global_virtual_clock, self.testing_history['perf'][self.round]['top_1'],
                                 self.testing_history['perf'][self.round]['top_5'], self.testing_history['perf'][self.round]['loss'],
                                 self.testing_history['perf'][self.round]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
                self.log_test_result()

            self.test_model_id = (self.test_model_id + 1) % len(self.probs)
            self.broadcast_events_queue.append(commons.START_ROUND if self.test_model_id == 0 else commons.MODEL_TEST)

    def broadcast_aggregator_events(self, event):
        """Issue tasks (events) to aggregator worker processes"""
        self.broadcast_events_queue.append(event)

    def dispatch_client_events(self, event, clients=None):
        """Issue tasks (events) to clients"""
        if clients is None:
            clients = self.sampled_executors

        for client_id in clients:
            self.individual_client_events[client_id].append(event)

    def get_client_conf(self, clientId):
        """Training configurations that will be applied on clients"""
        conf = {
            'learning_rate': self.args.learning_rate,
            'model': None  # none indicates we are using the global model
        }
        return conf

    def create_client_task(self, executorId):
        """Issue a new client training task to the executor"""

        next_clientId = self.resource_manager.get_next_task(executorId)
        train_config = None
        # NOTE: model = None then the executor will load the global model broadcasted in UPDATE_MODEL
        model = None
        if next_clientId != None:
            model = self.mapped_models[next_clientId]
            config = self.get_client_conf(next_clientId)
            train_config = {'client_id': next_clientId, 'task_config': config}
        return train_config, model

    def get_test_config(self, client_id):
        """FL model testing on clients"""

        return {'client_id': client_id}, self.test_model_id

    def get_global_model(self):
        """Get global model that would be used by all FL clients (in default FL)"""
        return self.model

    def get_shutdown_config(self, client_id):
        return {'client_id': client_id}

    def add_event_handler(self, executor_id, client_id, event, meta, data):
        """ Due to the large volume of requests, we will put all events into a queue first."""
        self.sever_events_queue.append((executor_id, client_id, event, meta, data))

    def CLIENT_REGISTER(self, request, context):
        """FL Client register to the aggregator"""

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id uses the same executor_id (VMs) in simulations
        executor_id = request.executor_id
        executor_info = self.deserialize_response(request.executor_info)
        if executor_id not in self.individual_client_events:
            # logging.info(f"Detect new client: {executor_id}, executor info: {executor_info}")
            self.individual_client_events[executor_id] = collections.deque()
        else:
            logging.info(f"Previous client: {executor_id} resumes connecting")

        # We can customize whether to admit the clients here
        self.executor_info_handler(executor_id, executor_info)
        dummy_data = self.serialize_response(commons.DUMMY_RESPONSE)

        return job_api_pb2.ServerResponse(event=commons.DUMMY_EVENT,
                                          meta=dummy_data, data=dummy_data)

    def CLIENT_PING(self, request, context):
        """Handle client requests"""

        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            current_event = self.individual_client_events[executor_id].popleft(
            )
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    executor_id)
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].appendleft(
                            commons.CLIENT_TRAIN)
            elif current_event == commons.MODEL_TEST:
                response_msg, response_data = self.get_test_config(int(executor_id))
            elif current_event == commons.UPDATE_MODEL:
                response_data = self.get_global_model()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(int(executor_id))

        if current_event != commons.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")
        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        return job_api_pb2.ServerResponse(event=current_event,
                                          meta=response_msg, data=response_data)

    def CLIENT_EXECUTE_COMPLETION(self, request, context):
        """FL clients complete the execution task."""

        executor_id, client_id, event = request.executor_id, request.client_id, request.event
        execution_status, execution_msg = request.status, request.msg
        meta_result, data_result = request.meta_result, request.data_result

        if event == commons.CLIENT_TRAIN:
            # Training results may be uploaded in CLIENT_EXECUTE_RESULT request later,
            # so we need to specify whether to ask client to do so (in case of straggler/timeout in real FL).
            if execution_status is False:
                logging.error(f"Executor {executor_id} fails to run client {client_id}, due to {execution_msg}")
            if self.resource_manager.has_next_task(executor_id):
                # NOTE: we do not pop the train immediately in simulation mode,
                # since the executor may run multiple clients
                self.individual_client_events[executor_id].appendleft(
                    commons.CLIENT_TRAIN)

        elif event in (commons.MODEL_TEST, commons.UPLOAD_MODEL):
            self.add_event_handler(
                executor_id, client_id, event, meta_result, data_result)
        else:
            logging.error(f"Received undefined event {event} from client {client_id}")
        return self.CLIENT_PING(request, context)

    def event_monitor(self):
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
                executor_id, client_id, current_event, meta, data = self.sever_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(
                        self.deserialize_response(data), int(client_id))
                    if len(self.stats_util_accumulator) == sum(self.tasks_round):
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        self.deserialize_response(data))

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)

    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)


if __name__ == "__main__":
    aggregator = Aggregator(args)
    aggregator.run()
