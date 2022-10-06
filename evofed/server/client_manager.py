from fedscale.core.client_manager import clientManager
from evofed.client.client_perf import EvoFed_Perf_Client

from typing import Dict

class Evofed_ClientManager(clientManager):
    def register_client(self, hostId: int, clientId: int, size: int, speed: Dict[str, float], duration: float = 1) -> None:
        uniqueId = self.getUniqueId(hostId, clientId)
        user_trace = None if self.user_trace is None else self.user_trace[self.user_trace_keys[int(
            clientId) % len(self.user_trace)]]

        self.Clients[uniqueId] = EvoFed_Perf_Client(hostId, clientId, speed, user_trace)

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(clientId)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward': min(size, self.args.local_steps*self.args.batch_size),
                             'duration': duration,
                             }
                self.ucbSampler.register_client(clientId, feedbacks=feedbacks)
        else:
            del self.Clients[uniqueId]

    def getCompletionTime(self, clientId, batch_size, upload_step, upload_size, download_size, model_flops=None):
        return self.Clients[self.getUniqueId(0, clientId)].getCompletionTime(
            batch_size=batch_size, upload_step=upload_step,
            upload_size=upload_size, download_size=download_size,
            model_flops=model_flops
        )
