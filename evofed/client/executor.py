from fedscale.core.execution.executor import Executor
from fedscale.utils.model_test_module import test_model
from fedscale.core.fllibs import tokenizer, parser
from fedscale.dataloaders.divide_data import select_dataset
from evofed.client.client import EvoFed_Client
import pickle, time, torch, logging, gc

class EvoFed_Executor(Executor):
    def __init__(self, args):
        super().__init__(args)
        self.model_num = 0
    
    def update_model_handler(self, model):
        self.round += 1
        assert(isinstance(model, list))
        for i, m in enumerate(model):
            with open(f"{self.temp_model_path}_{i}", 'wb') as model_out:
                pickle.dump(m, model_out)
        self.model_num = len(model)

    def load_global_model(self, model_id):
        """ Load last global model

        Returns:
            PyTorch or TensorFlow model: The lastest global model

        """
        with open(f"{self.temp_model_path}_{model_id}", 'rb') as model_in:
            model = pickle.load(model_in)
        return model

    def get_client_trainer(self, conf):
        return EvoFed_Client(conf)
    
    def training_handler(self, clientId, conf, model=None):
        """Train model given client id
        
        Args:
            clientId (int): The client id.
            conf (dictionary): The client runtime config.

        Returns:
            dictionary: The train result
        
        """
        # load last global model
        client_model = self.load_global_model(conf.model_id)
        # disable RL
        conf.clientId, conf.device = clientId, self.device
        conf.tokenizer = tokenizer
        client_data = select_dataset(clientId, self.training_sets,
                                        batch_size=conf.batch_size, args=self.args,
                                        collate_fn=self.collate_fn
                                        )

        client = self.get_client_trainer(conf)
        train_res = client.train(
            client_data=client_data, model=client_model, conf=conf)

        logging.info(f'training of model {conf.model_id} at client {conf.clientId} completed in executor {self.this_rank}')

        train_res['model_id'] = conf.model_id

        return train_res

    def testing_handler(self, args, config=None):
        """Test model
        
        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py
            config (dictionary): Variable arguments from coordinator.
        Returns:
            dictionary: The test result

        """
        evalStart = time.time()
        device = self.device
        results = {}
        for model_id in range(self.model_num):
            model = self.load_global_model(model_id)

            data_loader = select_dataset(self.this_rank, self.testing_sets,
                                            batch_size=args.test_bsz, args=args,
                                            isTest=True, collate_fn=self.collate_fn
                                            )

            if self.task == 'voice':
                from torch_baidu_ctc import CTCLoss
                criterion = CTCLoss(reduction='mean').to(device=device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(device=device)

            test_res = test_model(self.this_rank, model, data_loader,
                                    device=device, criterion=criterion, tokenizer=tokenizer)

            test_loss, acc, acc_5, testResults = test_res
            logging.info("After aggregation round {} of model {}, CumulTime {}, eval_time {}, test_loss {}, test_accuracy {:.2f}%, test_5_accuracy {:.2f}% \n"
                            .format(self.round, model_id, round(time.time() - self.start_run_time, 4), round(time.time() - evalStart, 4), test_loss, acc*100., acc_5*100.))

            results[model_id] = testResults
            gc.collect()

        return results
    
if __name__ == '__main__':
    executor = EvoFed_Executor(parser.args)
    executor.run()