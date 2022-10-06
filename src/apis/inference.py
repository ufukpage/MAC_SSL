
import torch
import os
import mmcv

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from .env import get_root_logger
from .test import multi_gpu_test, single_gpu_test
from ..datasets.dataloader import build_dataloader
import numpy as np
import matplotlib.pyplot as plt
from mmcv.runner import get_dist_info


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = np.array(cm.tolist())
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white"
                        if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test_network(model,
                 dataset,
                 cfg,
                 distributed=False,
                 logger=None,
                 progress=False):
    if logger is None:
        logger = get_root_logger(cfg.log_level)
    logger.info(f"Ckpt path: {cfg.checkpoint}")
    if cfg.checkpoint.endswith('.pth'):
        prefix = os.path.basename(cfg.checkpoint)[:-4]
    else:
        prefix = 'unspecified'

    out_name = f'eval_{dataset.__class__.__name__}_{dataset.name}'
    output_dir = os.path.join(cfg.work_dir, out_name)
    mmcv.mkdir_or_exist(output_dir)

    cache_path = os.path.join(output_dir, f'{prefix}_results.pkl')
    if os.path.isfile(cache_path):
        logger.info(f"Load results from {cache_path}")
        results = mmcv.load(cache_path)
    else:
        load_checkpoint(model, cfg.checkpoint, logger=logger)
        # save config in pt_model_3
        model.cfg = cfg
        # build dataloader
        multiprocessing_context = None
        if cfg.get('numpy_seed_hook', True) and cfg.data.workers_per_gpu > 0:
            multiprocessing_context = 'spawn'
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            multiprocessing_context=multiprocessing_context
        )

        # start training
        if distributed:
            if cfg.get('syncbn', False):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False
            )
            results = multi_gpu_test(model, data_loader, progress=progress)
        else:
            device_ids = list(range(cfg.gpus))
            model = MMDataParallel(model, device_ids=device_ids).cuda()
            results = single_gpu_test(model, data_loader)

        mmcv.dump(results, cache_path)

    # evaluate results
    eval_results = dataset.evaluate(results, logger)
    mmcv.dump(eval_results, os.path.join(output_dir,
                                         f'{prefix}_eval_results.json'))

    rank, world_size = get_dist_info()
    if rank == 0:
        # evaluate results
        eval_results = dataset.evaluate(results, logger, output_dir=output_dir)
        mmcv.dump(eval_results, os.path.join(output_dir,
                                             f'{prefix}_eval_results.json'))

        confusion_matrix = mmcv.load(os.path.join(output_dir, f'confusion_matrix.pkl'))

        labels = np.array([dataset.data_source[_]['name'].split("/")[0]
                           for _ in range(len(dataset))])

        labels = np.unique(labels)
        plt.figure(figsize=(len(labels), len(labels)))

        plot_confusion_matrix(confusion_matrix, labels, normalize=True, title='Confusion Matrix', cmap=plt.cm.Blues)
        plt.savefig(os.path.join(output_dir, f'confusion_matrix.png'))

        import csv
        with open(os.path.join(output_dir, 'confusion_matrix.csv'), 'w', encoding='UTF8', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            writer.writerow(labels)
            # write a row to the csv file
            for row in confusion_matrix:
                writer.writerow(row.tolist())
