from fedscale.core.internal.client import Client

base_flops = 7446000.0

class EvoFed_Perf_Client(Client):
    def getCompletionTime(self, batch_size, upload_step, upload_size, download_size, model_flops: float, augmentation_factor=3):
        if not model_flops:
            model_flops = base_flops
        speed = self.compute_speed * model_flops /  base_flops
        return {'computation': augmentation_factor * batch_size * upload_step*float(speed)/1000.,
                'communication': (upload_size+download_size)/float(self.bandwidth)}
