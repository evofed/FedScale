# package for aggregator
from fedscale.core.fllibs import *
import csv

logDir = os.path.join(args.log_path, "logs", args.job_name,
                      args.time_stamp, 'aggregator')
logFile = os.path.join(logDir, 'log')

statsDir = os.path.join(logDir, "stats")
statsFile = os.path.join(statsDir, "aggregated.csv")
statsHeader = ["round", "num_model", "num_converging", "num_converged", "average_loss", "average_test_accuracy", "tmstp_on_completion"]
modelDir = os.path.join(statsDir, "models")
modelStatsHeader = ["round", "trained_round", "loss", "average_test_accuracy"]
clientDir = os.path.join(statsDir, "clients")
clientStatsHeader = ["round", "trained_model", "utility", "best_model", "test_accuracy"]


def write_aggregated_stats(num_round: int, num_model: int, num_converging: int, 
                     num_converged: int, average_loss: int, average_test_accuracy: float, 
                     tmstp_on_completion: float):
    with open(statsFile, "a") as f:
        writer = csv.writer(f)
        writer.writerow([num_round, num_model, num_converging, num_converged, 
                         average_loss, average_test_accuracy, tmstp_on_completion])

def writer_model_stats(model_id: int, num_round: int, trained_round: int, 
                       loss: float, average_test_accuracy: float):
    modelFile = os.path.join(modelDir, f"{model_id}.csv")
    if not os.path.exists(modelFile):
        with open(modelFile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(modelStatsHeader)
    with open(modelFile, "a") as f:
        writer = csv.writer(f)
        writer.writerow([num_round, trained_round, loss, average_test_accuracy])
    
def write_client_stats(client_id: int, num_round: int, trained_model: int, utility: float, 
                       best_model: int, test_accuracy: float):
    clientFile = os.path.join(clientDir, f"{client_id}.csv")
    if not os.path.exists(clientFile):
        with open(clientFile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(clientStatsHeader)
    with open(clientFile, "a") as f:
        writer = csv.writer(f)
        writer.writerow(([num_round, trained_model, utility, best_model, test_accuracy]))
        

def init_logging():
    if not os.path.isdir(logDir):
        os.makedirs(logDir, exist_ok=True)
    
    if not os.path.isdir(statsDir):
        os.makedirs(statsDir, exist_ok=True)

    if not os.path.isdir(modelDir):
        os.makedirs(modelDir, exist_ok=True)

    if not os.path.isdir(clientDir):
        os.makedirs(clientDir, exist_ok=True)

    with open(statsFile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(statsHeader)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='(%m-%d) %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logFile, mode='a'),
            logging.StreamHandler()
        ])


def initiate_aggregator_setting():
    init_logging()

def aggregate_test_result(test_result_accumulator, task, round_num, global_virtual_clock, testing_history):    
    
    accumulator = test_result_accumulator[0]
    for i in range(1, len(test_result_accumulator)):
        if task == "detection":
            for key in accumulator:
                if key == "boxes":
                    for j in range(596):
                        accumulator[key][j] = accumulator[key][j] + \
                            test_result_accumulator[i][key][j]
                else:
                    accumulator[key] += test_result_accumulator[i][key]
        else:
            for key in accumulator:
                accumulator[key] += test_result_accumulator[i][key]
    if task == "detection":
        testing_history['perf'][round_num] = {'round': round_num, 'clock': global_virtual_clock,
                                                    'top_1': round(accumulator['top_1']*100.0/len(test_result_accumulator), 4),
                                                    'top_5': round(accumulator['top_5']*100.0/len(test_result_accumulator), 4),
                                                    'loss': accumulator['test_loss'],
                                                    'test_len': accumulator['test_len']
                                                    }
    else:
        testing_history['perf'][round_num] = {'round': round_num, 'clock': global_virtual_clock,
                                                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                                                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                                                    'loss': accumulator['test_loss']/accumulator['test_len'],
                                                    'test_len': accumulator['test_len']
                                                    }

    logging.info("FL Testing in round: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                    .format(round_num, global_virtual_clock, testing_history['perf'][round_num]['top_1'],
                            testing_history['perf'][round_num]['top_5'], testing_history['perf'][round_num]['loss'],
                            testing_history['perf'][round_num]['test_len']))


initiate_aggregator_setting()
