import os, torch, json, sklearn, seaborn, omegaconf
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything

from solo.utils.misc import make_contiguous
from solo.utils.auto_resumer import AutoResumer
from solo.data.dali_dataloader import ClassificationDALI
from solo.methods.mocov2_disentangle import MocoDisentangle
from solo.utils.test_utils import quadratic_weighted_kappa_


class MocoDisentangle_manipulate(MocoDisentangle):
    def __init__(self, cfg: omegaconf.DictConfig, Pure: bool):
        super(MocoDisentangle_manipulate, self).__init__(cfg, Pure)
        self.eval()

    @torch.no_grad()
    def obtain_feats_pure(self, X, target):
        if not self.no_channel_last: X = X.to(memory_format=torch.channels_last)
        # feats = self.backbone(X)
        feats = torch.flatten(torch.nn.AdaptiveAvgPool2d((1, 1))(self.backbone(X)), 1)
        output = {'feats': feats, 'targets': target}
        return output

    @torch.no_grad()
    def obtain_logit_pure(self, output:dict, key:str):
        logits  = self.classifier(output[key].to(self.device))
        output.update({'logits': logits, })
        return output

    @torch.no_grad()
    def obtain_decouple_info(self, feats_map, target):
        feats_map = feats_map.to(self.device)
        fusion = self.linear(self.encoder(feats_map.type(torch.float32)))
        cls_mu, cls_sigma, sps_mu, sps_sigma = torch.chunk(fusion, 4, dim=1)
        cls_mu, cls_sigma = self.scale_cls(cls_mu, cls_sigma)
        sps_mu, sps_sigma = self.scale_sps(sps_mu, sps_sigma)

        Z_cls = torch.randn_like(cls_sigma) * torch.exp(0.5 * cls_sigma) + cls_mu
        Z_sps = torch.randn_like(sps_sigma) * torch.exp(0.5 * sps_sigma) + sps_mu

        feats_rs = self.decode(Z_cls, Z_sps)
        output={
            'cls_mu': cls_mu, 'cls_sigma': cls_sigma,
            'sps_mu': sps_mu, 'sps_sigma': sps_sigma,
            'feats_rs': feats_rs, 'targets_rs': target,
        }
        return output

    @torch.no_grad()
    def decode(self, Z_cls, Z_sps):
        Z_cls, Z_sps = Z_cls.to(self.device), Z_sps.to(self.device)
        _, L = Z_cls.shape
        for indx in [0,2,4,6]:
            Z_sps = self.decoder[indx+1](self.decoder[indx](Z_sps))
            Z_sps = Z_sps + torch.nn.functional.interpolate(Z_cls.unsqueeze(1), scale_factor=Z_sps.shape[1]//L).squeeze(1)
        for indxs in range(8,14):
            Z_sps = self.decoder[indxs](Z_sps)
        return Z_sps

    @torch.no_grad()
    def manipulate(self, current_feas, current_lbl, arm_lbl):
        current_feas = current_feas.to(self.device)
        fusion = self.linear(self.encoder(current_feas.type(torch.float32)))
        cls_mu, cls_sigma, sps_mu, sps_sigma = torch.chunk(fusion, 4, dim=1)
        cls_mu, cls_sigma = self.scale_cls(cls_mu, cls_sigma)
        sps_mu, sps_sigma = self.scale_sps(sps_mu, sps_sigma)

        while arm_lbl!=current_lbl:
            if  arm_lbl>current_lbl:
                cls_mu = cls_mu + self.casual_mu[current_lbl]
                cls_sigma = cls_sigma + self.casual_sg[current_lbl]
                current_lbl = current_lbl + 1
            else:
                current_lbl = current_lbl - 1
                cls_mu = cls_mu - self.casual_mu[current_lbl]
                cls_sigma = cls_sigma - self.casual_sg[current_lbl]

        Z_cls = torch.randn_like(cls_sigma) * torch.exp(0.5 * cls_sigma) + cls_mu
        Z_sps = torch.randn_like(sps_sigma) * torch.exp(0.5 * sps_sigma) + sps_mu

        feats_rs = self.decode(Z_cls, Z_sps)

        return feats_rs

    @torch.no_grad()
    def sweap(self, swq1, swq2):
        swq1, swq2 = swq1.to(self.device), swq2.to(self.device)
        swq = torch.cat([swq1,swq2], dim=0).type(torch.float32)
        fusion = self.linear(self.encoder(swq))
        fusion1, fusion2 = torch.chunk(fusion, 2, dim=0)

        cls_mu1, cls_sigma1, sps_mu1, sps_sigma1 = torch.chunk(fusion1, 4, dim=1)
        cls_mu1, cls_sigma1 = self.scale_cls(cls_mu1, cls_sigma1)
        sps_mu1, sps_sigma1 = self.scale_sps(sps_mu1, sps_sigma1)

        Z_cls1 = torch.randn_like(cls_sigma1) * torch.exp(0.5 * cls_sigma1) + cls_mu1
        Z_sps1 = torch.randn_like(sps_sigma1) * torch.exp(0.5 * sps_sigma1) + sps_mu1

        cls_mu2, cls_sigma2, sps_mu2, sps_sigma2 = torch.chunk(fusion2, 4, dim=1)
        cls_mu2, cls_sigma2 = self.scale_cls(cls_mu2, cls_sigma2)
        sps_mu2, sps_sigma2 = self.scale_sps(sps_mu2, sps_sigma2)

        Z_cls2 = torch.randn_like(cls_sigma2) * torch.exp(0.5 * cls_sigma2) + cls_mu2
        Z_sps2 = torch.randn_like(sps_sigma2) * torch.exp(0.5 * sps_sigma2) + sps_mu2

        feats_2rs1 = self.decode(Z_cls2, Z_sps1)
        feats_1rs2 = self.decode(Z_cls1, Z_sps2)

        return feats_1rs2, feats_2rs1

class AdvancedTest():
    def __init__(self, cfg, model:MocoDisentangle_manipulate, modal:str='Train', select_num:int=30):
        print('Task name: {}'.format(cfg.name))
        self.cfg = cfg
        self.select_num = select_num
        self.model, self.modal = model, modal
        self.color = seaborn.color_palette("hls", 5)
        self.legend = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        train_loader = ClassificationDALI(cfg, cfg.data.train_path).test_dataloader()
        test_loader = ClassificationDALI(cfg, cfg.data.val_path).test_dataloader()
        self.loader = train_loader if modal == 'Train' else test_loader
        self.task_name = 'task: ' + self.cfg.name + ' task_id: ' + self.cfg.wandb_run_id

    def base_process(self, ckpt_file):
        print('Process checkpoints: {}'.format(str(ckpt_file)))
        weight = torch.load(ckpt_file)['state_dict']
        self.model.load_state_dict(weight, strict=False)
        make_contiguous(self.model)
        self.model.to(memory_format=torch.channels_last).cuda()

        feats, targets = [], []
        with torch.no_grad():
            for data, target, index in self.loader:
                with torch.cuda.amp.autocast(): kwargs: dict = self.model.obtain_feats_pure(data, target)
                targets.append(target.detach().cpu())
                feats.append(kwargs['feats'].detach().cpu())
        train_output = {'targets': torch.cat(targets, dim=0), 'feats': torch.cat(feats, dim=0)}
        return train_output

    def test_process(self, output:dict, key:str):
        output = self.model.obtain_logit_pure(output, key)
        self.test_fn(output['logits'], output['targets'])

    def swap_process(self, output:dict, key:str, sweap_id1:int, sweap_id2:int):
        data_A, data_B = output[key][output['targets']==sweap_id1], output[key][output['targets']==sweap_id2]
        Y_A, Y_B = output['targets'][output['targets']==sweap_id1], output['targets'][output['targets']==sweap_id2]
        with torch.no_grad():
            with torch.cuda.amp.autocast(): sweap_A, sweap_B = self.model.sweap(data_A, data_B)
        output.update({
            'feats_swp': torch.cat([data_A, data_B], dim=0).detach().cpu(),
            'sweap_swp': torch.cat([sweap_A, sweap_B], dim=0).detach().cpu(),
            'targets_swp': torch.cat([Y_A, Y_B], dim=0).detach().cpu(),
        })
        return output

    def manipulate_process(self, output:dict, key:str, cur_lbl:int, arm_lbl:int):
        data = output[key][output['targets']==cur_lbl]
        with torch.no_grad():
            with torch.cuda.amp.autocast(): data_mpl = self.model.manipulate(data, cur_lbl, arm_lbl)
        output.update({
            'feats_ogn': torch.cat([
                output[key][output['targets']==cur_lbl],
                output[key][output['targets']==arm_lbl],
                ], dim=0).detach().cpu(),
            'targets_ogn': torch.cat([
                output['targets'][output['targets']==cur_lbl],
                output['targets'][output['targets']==arm_lbl]
            ], dim=0).detach().cpu(),
            'feats_mpl': data_mpl.detach().cpu(),
            'targets_mpl': output['targets'][output['targets']==arm_lbl].detach().cpu(),
        })
        return output

    def forward_process(self, output:dict, key:str, gap:list):
        temp_data = {}
        with torch.no_grad():
            for elem in gap:
                data = output[key][output['targets']==elem]
                Y = output['targets'][output['targets']==elem]
                with torch.cuda.amp.autocast(): output_kw = self.model.obtain_decouple_info(data, Y)
                for k, value in output_kw.items():
                    if k not in temp_data: temp_data[k] = []
                    temp_data[k].append(value)
            for key, value in temp_data.items(): temp_data[key] = torch.cat(value, dim=0).detach().cpu()
            output.update(temp_data)
        return output

    def filter(self, out, num_samples, num_classes):
        points, index = [num_samples, ] * num_classes, []
        for i, lbl in enumerate(out['target']):
            if points[lbl] > 0:
                points[lbl] = points[lbl] - 1
                index.append(i)

        output = {}
        for key in out.keys():
            temp = []
            for indx in index: temp.append(out[key][indx].unsqueeze(0))
            output[key] = torch.cat(temp, dim=0)
        return output

    def test_fn(self, logit_matrix:torch.tensor, label:torch.tensor):
        predict = logit_matrix.argmax(dim=1)
        matrix = sklearn.metrics.confusion_matrix(label.numpy(), predict.numpy())
        QWK = quadratic_weighted_kappa_(label.numpy(), predict.numpy())
        VA = np.average(matrix.diagonal() / matrix.sum(axis=1))
        SVA = matrix.diagonal() / matrix.sum(axis=1)
        print('{} average accuracy: {}; QWK: {}; special accuracy: {}'.format(self.modal, VA, QWK, SVA))
        print(matrix)

    def select_fn(self, data:torch.tensor, Y:torch.tensor):
        if self.select_num:
            Y, data = Y.numpy(), data.numpy()
            TT = sklearn.metrics.silhouette_samples(data, Y, metric='euclidean')
            ABC = np.column_stack((Y, TT, data))
            sorted_ABC = ABC[ABC[:, 1].argsort()]

            sdata, sY = [], []
            for Y_value in range(5):
                Y_indices = np.where(sorted_ABC[:, 0] == Y_value)[0]
                Y_TT, Y_DT = sorted_ABC[:, 0][Y_indices], sorted_ABC[:, 2:][Y_indices]
                sdata.extend(list(Y_DT[-self.select_num:]))
                sY.extend(list(Y_TT[-self.select_num:]))
            data, Y = np.array(sdata), np.array(sY)
        else: Y, data = data.numpy(), Y.numpy()
        return torch.tensor(data), torch.tensor(Y)

    def gap_fn(self, data:torch.tensor, Y:torch.tensor, gap:list):
        data, Y, tdata, tY = data.numpy(), Y.numpy(), [], []
        for elem in gap:
            indices = np.where(Y == elem)[0]
            tdata.extend(data[indices])
            tY.extend(Y[indices])
        data, Y = np.array(tdata), np.array(tY)
        return torch.tensor(data), torch.tensor(Y)

    def vis_fn(
            self,
            data:torch.tensor,
            Y:torch.tensor,
            gap:list,
            data_en:torch.tensor=None,
            Y_en:torch.tensor=None,
            gap_en: list=None,
            name:str=None
        ):
        data, Y, ptr = data.numpy(), Y.numpy(), data.shape[0]
        color, legends = [self.color[i] for i in gap], [self.legend[i] for i in gap]
        if not data_en is  None and not Y_en is  None and not gap_en is None:
            data_en, Y_en = data_en.numpy(), Y_en.numpy()
            data = np.vstack((data, data_en))

        tsne_data = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        data = tsne_data[0:ptr,:]

        for class_id, lbl, clr in zip(gap, legends, color):
            data_class = data[Y == class_id]
            ax.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='o', label=lbl)

        if not data_en is None and not Y_en is None and not gap_en is None:
            data_en = tsne_data[ptr:, :]
            color_en, legends_en = [self.color[i] for i in gap_en], [self.legend[i] + '_enhance' for i in gap_en]
            for class_id, lbl, clr in zip(gap_en, legends_en, color_en):
                data_class = data_en[Y_en == class_id]
                ax.scatter(data_class[:, 0], data_class[:, 1], color=clr, marker='+', label=lbl)

        ax.legend()
        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        plt.title(self.task_name + name)
        plt.show()

def main(path_root, decouple:bool=False, modal:str='last', seed:int=0):
    dir_root = Path(path_root)
    with open(dir_root / 'args.json') as user_file: file_contents = user_file.read()
    cfg = OmegaConf.create(json.loads(file_contents))
    seed_everything(seed)

    model = MocoDisentangle_manipulate(cfg, decouple)
    auto_resumer = AutoResumer(checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method), max_hours=100000)
    resume_from_checkpoint, wandb_id = auto_resumer.find_checkpoint(cfg)
    if wandb_id is None: dir_path = dir_root
    else: dir_path = Path(os.path.join(cfg.checkpoint.dir, cfg.method, wandb_id))

    if modal=='last':
        ckpt_file = [dir_path / 'last.ckpt']
    elif modal=='best':
        ckpt_file = [dir_path / f for f in os.listdir(dir_path) if (f.endswith(".ckpt") and 'Best' in f)]
    else:
        ckpt_file = [dir_path / 'Z_frequency' / f for f in os.listdir(dir_path / 'Z_frequency') if (f.endswith(".ckpt"))]
        # ckpt_file = [str(i) for i in ckpt_file].sort()

    for ckpt_file_i in ckpt_file: work_process_real_time(cfg, model, ckpt_file_i)

def work_process_real_time(cfg, model, ckpt_file_i):
    core = AdvancedTest(cfg, model, modal='Train')
    train_output = core.base_process(ckpt_file_i)

    work_set = {'feats': train_output['feats'], 'targets': train_output['targets']}
    # core.vis_fn(work_set['feats'], work_set['targets'], gap=[0, 1, 2, 3, 4], name=' Feats')
    work_set = core.forward_process(work_set, 'feats', gap=[0, 1, 2, 3, 4])
    # core.vis_fn(work_set['feats_rs'], work_set['targets_rs'], gap=[0, 1, 2, 3, 4], name=' Feats_RS')
    # core.vis_fn(work_set['cls_mu'], work_set['targets_rs'], gap=[0, 1, 2, 3, 4], name=' Mu-CLS')
    # core.vis_fn(work_set['sps_mu'], work_set['targets_rs'], gap=[0, 1, 2, 3, 4], name=' Mu-SPS')
    print('D')
    torch.save(work_set, 'total_data_1.dat')

    data, Y = core.select_fn(train_output['feats'], train_output['targets'])
    work_set_part = {'feats': data, 'targets': Y}
    work_set_part = core.forward_process(work_set_part, 'feats', gap=[0, 1, 2, 3, 4])
    core.vis_fn(work_set_part['feats_rs'], work_set_part['targets_rs'], gap=[0, 1, 2, 3, 4], name=' Feats_RS')
    core.vis_fn(work_set_part['cls_mu'], work_set_part['targets_rs'], gap=[0, 1, 2, 3, 4], name=' Mu-CLS')
    core.vis_fn(work_set_part['sps_mu'], work_set_part['targets_rs'], gap=[0, 1, 2, 3, 4], name=' Mu-SPS')

    # core.vis_fn(
    #     data=work_set_part['feats'], Y=work_set_part['targets'], gap=[0, 1, 2, 3, 4],
    #     data_en=work_set_part['feats_rs'], Y_en=work_set_part['targets'], gap_en=[0, 1, 2, 3, 4], name=' RS_Com'
    # )

    work_set_part = core.swap_process(work_set_part, key='feats', sweap_id1=1, sweap_id2=3)
    # core.vis_fn(
    #     data=work_set_part['feats_rs'], Y=work_set_part['targets'], gap=[0, 1, 2, 3, 4],
    #     data_en=work_set_part['sweap_swp'], Y_en=work_set_part['targets_swp'], gap_en=[1, 3], name=' Swap'
    # )
    work_set_part = core.manipulate_process(work_set_part, key='feats', cur_lbl=1, arm_lbl=3)
    # core.vis_fn(
    #     data=work_set_part['feats_rs'], Y=work_set_part['targets'], gap=[0, 1, 2, 3, 4],
    #     data_en=work_set_part['feats_mpl'], Y_en=work_set_part['targets_mpl'], gap_en=[3], name=' Manipulate'
    # )
    # torch.save(work_set_part, 'work_set_part.dat')
    print('Done')

if __name__ == "__main__":
    data = torch.load('./work_set_part.dat')
    print('')

    # main('./trained_models/mocodisentangle/bry0ycf7', decouple=False, modal='Z_frequency') # 'Z_frequency'

    # 尝试
    # main('./trained_models/mocodisentangle/96y1vbz9', decouple=False, modal='last')
    # main('./trained_models/mocodisentangle/gdwg5yhj', decouple=False, modal='last')
    # main('./trained_models/mocodisentangle/z6ikdbz5', decouple=False, modal='last')
    main('./trained_models/arvix/supcondisentangle/upefo9pn', decouple=False, modal='last')


    # weight1 = torch.load('./trained_models/mocodisentangle/4so0vdm1/last.ckpt')['state_dict']
    # weight2 = torch.load('./trained_models/arvix/mocov2/40p4maje/last.ckpt')['state_dict']
    
    # t1 = 0
    # A = {}
    # for key in weight1:
    #     if 'backbone' in key and 'momentum' not in key:
    #         A[key] = weight1[key]
    #         t1 = t1 + weight1[key].sum()
    # print(t1)
    # t2 = 0
    # B = {}
    # for key in weight2:
    #     if 'backbone' in key and 'momentum' not in key:
    #         B[key] = weight2[key]
    #         t2 = t2 + weight2[key].sum()
    # print(t2)
    # print('D')