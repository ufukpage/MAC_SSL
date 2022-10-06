
import torch
import os
import mmcv

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from .env import get_root_logger
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from ..datasets.dataloader import build_dataloader
import time
from .test import collect_results_cpu
from mmcv.runner import get_dist_info
from tqdm import tqdm

def multi_gpu_extract_features(model, data_loader, progress=True):
    """Extract and save features for train split, several clips per video."""
    model.eval()
    results = []
    labels = []
    indices = []
    dataset = data_loader.dataset

    rank, world_size = get_dist_info()
    if rank == 0 and progress:
        prog_bar = mmcv.ProgressBar(len(dataset))

    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        label = data["gt_labels"].data[0].cuda()
        imgs = data['imgs'].data[0].cuda()
        with torch.no_grad():
            feat = model(imgs=imgs, retrieve_mode=True)
        results.extend(feat.cpu().numpy().tolist())
        # labels.append(label.repeat(1, 10).cpu().numpy().tolist())
        labels.extend(label.flatten().cpu().numpy().tolist())
        indices.extend(data['index'].data[0].flatten().cpu().numpy().tolist())
        if rank == 0 and progress:
            batch_size = len(feat)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    feats = collect_results_cpu(results, len(dataset), None)
    labels = collect_results_cpu(labels, len(dataset), None)
    indices = collect_results_cpu(indices, len(dataset), None)
    return feats, labels, indices


def single_gpu_extract_features(model, data_loader):
    """Extract and save features for train split, several clips per video."""
    model.eval()
    feats = []
    labels = []
    model = model.cuda()
    indices = []
    for i, data in enumerate(tqdm(data_loader)):
        label = data["gt_labels"].data[0].cuda()
        imgs = data['imgs'].data[0].cuda()
        with torch.no_grad():
            feat = model(imgs=imgs, retrieve_mode=True)
        feats.extend(feat.cpu().numpy().tolist())
        # labels.append(label.repeat(1, 10).cpu().numpy().tolist())
        labels.extend(label.flatten().cpu().numpy().tolist())
        indices.extend(data['index'].data[0].flatten().cpu().numpy().tolist())
        # if i == 10: break
    feats = np.array(feats)
    labels = np.array(labels)
    return feats, labels, np.array(indices)


def  topk_retrieval(X_train, y_train, X_test, y_test):
    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks}

    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], -1)

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        print('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

    return topk_correct


def retrieve(model,
             dataset,
             test_dataset,
             cfg,
             distributed=False,
             logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)
    logger.info(f"Ckpt path: {cfg.checkpoint}")
    if cfg.checkpoint.endswith('.pth'):
        prefix = os.path.basename(cfg.checkpoint)[:-4]
    else:
        prefix = 'unspecified'

    out_name = f'retrieve_{dataset.__class__.__name__}_{dataset.name}'
    output_dir = os.path.join(cfg.work_dir, out_name)
    mmcv.mkdir_or_exist(output_dir)

    multiprocessing_context = None
    if cfg.get('numpy_seed_hook', True) and cfg.data.workers_per_gpu > 0:
        multiprocessing_context = 'spawn'
    test_data_loader = build_dataloader(
        test_dataset,
        imgs_per_gpu=cfg.data.videos_per_gpu, #1, cfg.data.videos_per_gpu
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        multiprocessing_context=multiprocessing_context,
        drop_last=False
    )

    cache_path = os.path.join(output_dir, f'{prefix}_feats.pkl')
    label_cache_path = os.path.join(output_dir, f'{prefix}_labels.pkl')
    test_cache_path = os.path.join(output_dir, f'{prefix}_feats_test.pkl')
    test_label_cache_path = os.path.join(output_dir, f'{prefix}_labels_test.pkl')
    if os.path.isfile(cache_path):
        logger.info(f"Load features from {cache_path}")
        train_feats = mmcv.load(cache_path)
        train_labels = mmcv.load(label_cache_path)


        test_feats = mmcv.load(cache_path)
        test_labels = mmcv.load(label_cache_path)
    else:
        load_checkpoint(model, cfg.checkpoint, logger=logger)
        # save config in model
        model.cfg = cfg
        # build dataloader

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=cfg.data.videos_per_gpu, # 1, cfg.data.videos_per_gpu
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            multiprocessing_context=multiprocessing_context,
            drop_last=False
        )

        if distributed:
            if cfg.get('syncbn', False):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False
            )
            train_feats, train_labels, train_indices = multi_gpu_extract_features(model, data_loader)
            test_feats, test_label, test_indicess = multi_gpu_extract_features(model, test_data_loader)
        else:
            device_ids = list(range(cfg.gpus))
            model = MMDataParallel(model, device_ids=device_ids).cuda()
            train_feats, train_labels, train_indices = single_gpu_extract_features(model, data_loader)
            test_feats, test_labels, test_indices = single_gpu_extract_features(model, test_data_loader)

        mmcv.dump(train_feats, cache_path)
        mmcv.dump(train_labels, label_cache_path)

        mmcv.dump(train_indices, os.path.join(output_dir, f'{prefix}_indices.pkl'))

        mmcv.dump(test_feats, test_cache_path)
        mmcv.dump(train_labels, test_label_cache_path)
        mmcv.dump(test_indices, os.path.join(output_dir, f'{prefix}_indices_test.pkl'))


    # evaluate results
    retrieve_results = topk_retrieval(train_feats, train_labels, test_feats, test_labels)
    mmcv.dump(retrieve_results, os.path.join(output_dir,
                                         f'{prefix}_retrieve_results.json'))