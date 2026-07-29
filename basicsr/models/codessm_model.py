from collections import OrderedDict
from os import path as osp

import copy
import pyiqa
import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY.register()
class CoDeSSMModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Define the network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        # Define evaluation metric functions
        if self.opt['val'].get('metrics') is not None:
            self.metric_funcs = {}
            for _, m_opt in self.opt['val']['metrics'].items():
                mopt = m_opt.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                self.metric_funcs[name] = pyiqa.create_metric(
                    name,
                    device=self.device,
                    **mopt
                )

        # Load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()

        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            strict_load = self.opt['path'].get('strict_load', True)
            self.load_network(self.net_g, load_path, strict_load)

        if self.is_train:
            self.init_training_settings()

        self.net_g_best = copy.deepcopy(self.net_g)

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.net_g.train()

        # Define loss functions using build_loss from config
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # Set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        logger = get_root_logger()

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger.warning(f'Params {k} will not be optimized.')

        # Define optimizer
        optim_type = train_opt['optim_g'].pop('type')
        optim_class = getattr(torch.optim, optim_type)
        self.optimizer_g = optim_class(optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print(model)
        print("The number of parameters: {}".format(num_params))

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # CoDe-SSM supports return_aux=True to obtain the LHFM balance loss.
        self.output, aux_dict = self.net_g(self.lq, return_aux=True)

        l_g_total = 0
        loss_dict = OrderedDict()

        # 1. Pixel loss
        if self.cri_pix is not None:
            l_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_pix
            loss_dict['l_pix'] = l_pix

        # 2. FFT loss
        if self.cri_fft is not None:
            l_fft = self.cri_fft(self.output, self.gt)
            l_g_total += l_fft
            loss_dict['l_freq'] = l_fft

        # 3. LHFM balance loss.
        balance_loss = aux_dict.get('lhfm_balance_loss')
        if balance_loss is None:
            balance_loss = aux_dict.get('moe_balance_loss')

        if balance_loss is not None and torch.isfinite(balance_loss):
            balance_weight = self.opt['train'].get(
                'lhfm_balance_weight',
                self.opt['train'].get('moe_balance_weight', 0.01)
            )

            l_g_total += balance_weight * balance_loss
            loss_dict['l_lhfm_balance'] = balance_loss

        l_g_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()

        net_g = self.get_bare_model(self.net_g)

        # Use smaller min_size with limited GPU memory.
        min_size = 8000 * 8000

        lq_input = self.lq
        _, _, h, w = lq_input.shape

        if h * w < min_size:
            self.output = net_g.test(lq_input)
        else:
            self.output = net_g.test_tile(lq_input)

        if self.is_train:
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir=None):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')

        self.nondist_validation(
            dataloader,
            current_iter,
            epoch,
            tb_logger,
            save_img,
            save_as_dir
        )

    def nondist_validation(self, dataloader, current_iter, epoch, tb_logger, save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt.get('name', 'unknown')

        with_metrics = self.opt['val'].get('metrics') is not None

        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {
                    metric: 0
                    for metric in self.opt['val']['metrics'].keys()
                }

            self._initialize_best_metric_results(dataset_name)

            self.metric_results = {
                metric: 0
                for metric in self.metric_results
            }

            self.key_metric = self.opt['val'].get('key_metric')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            self.test()

            sr_img = tensor2img(self.output)

            if with_metrics:
                sr_tensor = img2tensor(sr_img).unsqueeze(0).to(self.device) / 255.0
                metric_data = [sr_tensor, self.gt]
            else:
                metric_data = None

            # Free GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'],
                        'image_results',
                        f'{current_iter}',
                        f'{img_name}.png'
                    )
                else:
                    suffix = self.opt['val'].get('suffix')

                    if suffix:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{suffix}.png'
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{self.opt["name"]}.png'
                        )

                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)

                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name in self.opt['val']['metrics'].keys():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if self.key_metric is not None:
                to_update = self._update_best_metric_result(
                    dataset_name,
                    self.key_metric,
                    self.metric_results[self.key_metric],
                    current_iter
                )

                if to_update:
                    for name in self.opt['val']['metrics'].keys():
                        self._update_metric_result(
                            dataset_name,
                            name,
                            self.metric_results[name],
                            current_iter
                        )

                    self.copy_model(self.net_g, self.net_g_best)
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)
            else:
                updated = []

                for name in self.opt['val']['metrics'].keys():
                    tmp_updated = self._update_best_metric_result(
                        dataset_name,
                        name,
                        self.metric_results[name],
                        current_iter
                    )
                    updated.append(tmp_updated)

                if sum(updated):
                    self.copy_model(self.net_g, self.net_g_best)
                    self.save_network(self.net_g, 'net_g_best', current_iter, epoch)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'

        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'

            if hasattr(self, 'best_metric_results'):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )

            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f'metrics/{dataset_name}/{metric}',
                    value,
                    current_iter
                )

    def get_current_visuals(self):
        vis_samples = 16

        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        out_dict['result'] = self.output.detach().cpu()[:vis_samples]

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]

        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter, epoch)
        self.save_training_state(epoch, current_iter)