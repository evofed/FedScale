from fedscale.core.execution.client import Client
from fedscale.core.fllibs import mask_tokens, tokenizer
import logging, math, torch
from torch.autograd import Variable
from ..lib.net2netlib import get_model_layer_grad, get_model_layer_weight


class EvoFed_Client(Client):

    def init_task(self, conf):
        if conf.task == "detection":
            self.im_data = Variable(torch.FloatTensor(1).cuda())
            self.im_info = Variable(torch.FloatTensor(1).cuda())
            self.num_boxes = Variable(torch.LongTensor(1).cuda())
            self.gt_boxes = Variable(torch.FloatTensor(1).cuda())

        self.epoch_train_loss = 1e-4
        self.completed_steps = 0
        self.loss_squre = 0
        self.layer_names = conf.layer_names

    def train(self, client_data, model, conf):
        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device

        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size)

        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None

        # NOTE: If one may hope to run fixed number of epochs, instead of iterations, 
        # use `while self.completed_steps < conf.local_steps * len(client_data)` instead
        while self.completed_steps < conf.local_steps:
            try:
                self.train_step(client_data, conf, model, optimizer, criterion)
            except Exception as ex:
                error_type = ex
                break

        state_dicts = model.state_dict()
        model_param = {p: state_dicts[p].data.cpu().numpy()
                       for p in state_dicts}
        results = {'clientId': clientId, 'moving_loss': self.epoch_train_loss,
                   'trained_size': self.completed_steps*conf.batch_size, 
                   'success': self.completed_steps == conf.local_steps}

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        # get layer gradient
        grad_dict = dict()
        for layer in self.layer_names:
            grad = get_model_layer_grad(model, layer[1])
            weight = get_model_layer_weight(model, layer[1])
            grad_dict[layer[1]] = torch.norm(grad.data.cpu()) / torch.norm(weight.data.cpu())
        results['gradient'] = grad_dict
        
        results['utility'] = math.sqrt(
            self.loss_squre)*float(trained_unique_samples)
        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results
    
    def train_step(self, client_data, conf, model, optimizer, criterion):

        for data_pair in client_data:
            if conf.task == 'nlp':
                (data, _) = data_pair
                data, target = mask_tokens(
                    data, tokenizer, conf, device=conf.device)
            elif conf.task == 'voice':
                (data, target, input_percentages,
                    target_sizes), _ = data_pair
                input_sizes = input_percentages.mul_(
                    int(data.size(3))).int()
            elif conf.task == 'detection':
                temp_data = data_pair
                target = temp_data[4]
                data = temp_data[0:4]
            else:
                (data, target) = data_pair

            if conf.task == "detection":
                self.im_data.resize_(data[0].size()).copy_(data[0])
                self.im_info.resize_(data[1].size()).copy_(data[1])
                self.gt_boxes.resize_(data[2].size()).copy_(data[2])
                self.num_boxes.resize_(data[3].size()).copy_(data[3])
            elif conf.task == 'speech':
                data = torch.unsqueeze(data, 1).to(device=conf.device)
            elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
                (data, masks) = data
                data, masks = Variable(data).to(
                    device=conf.device), Variable(masks).to(device=conf.device)

            else:
                data = Variable(data).to(device=conf.device)

            target = Variable(target).to(device=conf.device)

            if conf.task == 'nlp':
                outputs = model(data, labels=target)
                loss = outputs[0]
            elif conf.task == 'voice':
                outputs, output_sizes = model(data, input_sizes)
                outputs = outputs.transpose(0, 1).float()  # TxNxH
                loss = criterion(
                    outputs, target, output_sizes, target_sizes)
            elif conf.task == 'text_clf' and conf.model == 'albert-base-v2':
                outputs = model(
                    data, attention_mask=masks, labels=target)
                loss = outputs.loss
                output = outputs.logits
            elif conf.task == "detection":
                rois, cls_prob, bbox_pred, \
                    rpn_loss_cls, rpn_loss_box, \
                    RCNN_loss_cls, RCNN_loss_bbox, \
                    rois_label = model(
                        self.im_data, self.im_info, self.gt_boxes, self.num_boxes)

                loss = rpn_loss_cls + rpn_loss_box \
                    + RCNN_loss_cls + RCNN_loss_bbox

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls.item()
                loss_rcnn_box = RCNN_loss_bbox.item()
                
            else:
                output = model(data)
                loss = criterion(output, target)

            # ======== collect training feedback for other decision components [e.g., oort selector] ======

            if conf.task == 'nlp' or (conf.task == 'text_clf' and conf.model == 'albert-base-v2'):
                loss_list = [loss.item()]  # [loss.mean().data.item()]

            elif conf.task == "detection":
                loss_list = [loss.tolist()]
                loss = loss.mean()
            else:
                loss_list = loss.tolist()
                loss = loss.mean()

            temp_loss = sum(loss_list)/float(len(loss_list))
            self.loss_squre = sum([l**2 for l in loss_list]
                                )/float(len(loss_list))
            # only measure the loss of the first epoch
            if self.completed_steps < len(client_data):
                if self.epoch_train_loss == 1e-4:
                    self.epoch_train_loss = temp_loss
                else:
                    self.epoch_train_loss = (
                        1. - conf.loss_decay) * self.epoch_train_loss + conf.loss_decay * temp_loss

            # ========= Define the backward loss ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ========= Weight handler ========================
            self.optimizer.update_client_weight(
                conf, model, self.global_model if self.global_model is not None else None)

            self.completed_steps += 1

            if self.completed_steps == conf.local_steps:
                break