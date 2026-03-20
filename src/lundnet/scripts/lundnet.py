# This file is part of LundNet by F. Dreyer and H. Qu

"""
    lundnet.py: the entry point for LundNet.
"""

from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
import math
import copy

import tqdm
from functools import partial
import os, time, datetime, argparse, pickle

from lundnet.dgl_dataset import DGLGraphDatasetParticle, DGLGraphDatasetLund, collate_wrapper, collate_wrapper_tree
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def bkg_rejection_at_threshold(signal_eff, background_eff, sig_eff=0.5):
    """Background rejection at a given signal efficiency."""
    idx = np.argmin(np.abs(signal_eff - sig_eff)) + 1
    idx = min(idx, len(background_eff) - 1)
    return 1 / (1 - background_eff[idx])


def ROC_area(signal_eff, background_eff):
    """Area under the ROC curve."""
    normal_order = signal_eff.argsort()
    return np.trapz(background_eff[normal_order], signal_eff[normal_order])


def accuracy(preds, labels):
    """Return the accuracy."""
    if labels.ndim == 2:
        labels = labels[:, 1]
    return (preds.argmax(1) == labels).sum().item() / len(labels)


def epoch_auc_from_scores(labels, scores):
    """Compute AUC from binary labels and signal scores."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # evita crash se per qualche motivo c'è una sola classe
    if len(np.unique(labels)) < 2:
        return float('nan')

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eff_s = tpr
    eff_b = 1 - fpr
    return ROC_area(eff_s, eff_b)


def labels_to_numpy(labels):
    labels = labels.cpu().detach().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    if labels.ndim == 2:
        labels = labels[:, 1]
    return labels.astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--train-sig', type=str, default='')
    parser.add_argument('--train-bkg', type=str, default='')
    parser.add_argument('--val-sig', type=str, default='')
    parser.add_argument('--val-bkg', type=str, default='')
    parser.add_argument('--test-sig', type=str, default='')
    parser.add_argument('--test-bkg', type=str, default='')
    parser.add_argument('--model', type=str, default='lundnet5', choices=['lundnet5', 'lundnet2',
                                                                          'lundnet3', 'lundnet4',
                                                                          'particlenet', 'particlenet-lite'])
    parser.add_argument('--ln-kt-min', type=float, default=None)
    parser.add_argument('--ln-delta-min', type=float, default=None)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--test-output', type=str, default='')
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--nev', type=int, default=-1)
    parser.add_argument('--nev-val', type=int, default=-1)
    parser.add_argument('--nev-test', type=int, default=-1)
    parser.add_argument('--start-lr', type=float, default=0.001)
    parser.add_argument('--lr-steps', type=str, default='10,20')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max-depth', type=int, default=1000000)

    # nuovi argomenti per CV
    parser.add_argument('--cv-folds', type=int, default=0)
    parser.add_argument('--cv-validation-fraction', type=float, default=0.1)
    parser.add_argument('--cv-seed', type=int, default=12345)

    args = parser.parse_args()

    if 'lund' in args.model:
        from lundnet.JetTree import JetTree, LundCoordinates
        if args.model == 'lundnet4':
            LundCoordinates.change_dimension(4, ['lnz', 'lnDelta', 'lnKt', 'psi'])
        elif args.model == 'lundnet3':
            LundCoordinates.change_dimension(3, ['lnz', 'lnDelta', 'lnKt'])
        elif args.model == 'lundnet2':
            LundCoordinates.change_dimension(2, ['lnz', 'lnDelta'])
        kt_min = np.exp(args.ln_kt_min) if (args.ln_kt_min is not None and args.ln_kt_min > -99) else 0
        delta_min = np.exp(args.ln_delta_min) if args.ln_delta_min is not None else 0
        JetTree.change_cuts(kt_min, delta_min)
        print('Using %s, kt_min=%f and delta_min=%f' % (args.model, JetTree.ktmin, JetTree.deltamin))

    if args.demo:
        args.train_sig = 'sample_WW_500GeV.json.gz'
        args.train_bkg = 'sample_QCD_500GeV.json.gz'
        args.val_sig = 'sample_WW_500GeV.json.gz'
        args.val_bkg = 'sample_QCD_500GeV.json.gz'
        args.test_sig = 'sample_WW_500GeV.json.gz'
        args.test_bkg = 'sample_QCD_500GeV.json.gz'

    cv_mode = args.cv_folds and args.cv_folds > 1

    # training/testing mode
    if cv_mode:
        training_mode = True
    else:
        if args.train_bkg and args.train_sig:
            training_mode = True
        else:
            assert(args.load)
            training_mode = False

    # data format
    DGLGraphDataset = DGLGraphDatasetLund if 'lund' in args.model else DGLGraphDatasetParticle
    dataset_kwargs = {'max_depth': args.max_depth} if 'lund' in args.model else {}

    # model parameter
    if args.model == 'particlenet':
        from lundnet.ParticleNet import ParticleNet
        Net = ParticleNet
        conv_params = [
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
        ]
        fc_params = [(256, 0.1)]
        use_fusion = False
        if args.batch_size <= 0:
            args.batch_size = 256
        collate_fn = partial(collate_wrapper, k=conv_params[0][0])
    elif args.model == 'particlenet-lite':
        from lundnet.ParticleNet import ParticleNet
        Net = ParticleNet
        conv_params = [
            (7, (32, 32, 32)),
            (7, (64, 64, 64))
        ]
        fc_params = [(128, 0.1)]
        use_fusion = False
        if args.batch_size <= 0:
            args.batch_size = 1024
        collate_fn = partial(collate_wrapper, k=conv_params[0][0])
    else:
        from lundnet.LundNet import LundNet
        Net = LundNet
        conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]]
        fc_params = [(256, 0.1)]
        use_fusion = True
        if args.batch_size <= 0:
            args.batch_size = 256
        collate_fn = collate_wrapper_tree

    # device
    dev = torch.device(args.device)

    # load data (only in standard mode)
    if not cv_mode:
        if training_mode:
            train_data = DGLGraphDataset(args.train_bkg, args.train_sig, nev=args.nev, **dataset_kwargs)
            val_data = DGLGraphDataset(args.val_bkg, args.val_sig, nev=args.nev_val, **dataset_kwargs)
            train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                      collate_fn=collate_fn, shuffle=True, drop_last=True, pin_memory=True)
            val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                    collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
            input_dims = train_data.num_features
        else:
            test_data = DGLGraphDataset(args.test_bkg, args.test_sig, nev=args.nev_test, **dataset_kwargs)
            test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                     collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)
            input_dims = test_data.num_features

        # model
        model = Net(input_dims=input_dims, num_classes=2,
                    conv_params=conv_params,
                    fc_params=fc_params,
                    use_fusion=use_fusion)
        model = model.to(dev)

        num_total_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_buffers = sum(b.numel() for b in model.buffers())
        print("Total parameters:", num_total_params)
        print("Trainable parameters:", num_trainable_params)
        print("Buffers:", num_buffers)

    def train(model, opt, scheduler, train_loader, dev, loss_func):
        model.train()
        total_loss = 0.0
        num_batches = 0
        total_correct = 0
        count = 0
        tic = time.time()

        all_scores = []
        all_labels = []

        with tqdm.tqdm(train_loader, ascii=True) as tq:
            for batch in tq:
                label = batch.label
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()

                opt.zero_grad()
                logits = model(batch.batch_graph.to(dev), batch.features.to(dev))
                loss = loss_func(logits, label)
                loss.backward()
                opt.step()

                probs = torch.softmax(logits, dim=1)[:, 1]
                _, preds = logits.max(1)

                num_batches += 1
                count += num_examples
                loss_value = loss.item()
                correct = (preds == label).sum().item()

                total_loss += loss_value
                total_correct += correct

                all_scores.append(probs.detach().cpu().numpy())
                all_labels.append(label.detach().cpu().numpy())

                tq.set_postfix({
                    'Loss': '%.5f' % loss_value,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                })

        scheduler.step()

        ts = time.time() - tic
        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_correct / max(count, 1)

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        avg_auc = epoch_auc_from_scores(all_labels, all_scores)

        print('Trained over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(
            count=count, ts=ts, speed=count / ts
        ))

        return avg_loss, avg_acc, avg_auc

    def evaluate(model, test_loader, dev, loss_func=None,
             return_scores=False, return_time=False, return_metrics=False):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        num_batches = 0
        count = 0
        scores = []
        labels_all = []
        tic = time.time()

        with torch.no_grad():
            with tqdm.tqdm(test_loader, ascii=True) as tq:
                for batch in tq:
                    label = batch.label
                    num_examples = label.shape[0]
                    label = label.to(dev).squeeze().long()

                    logits = model(batch.batch_graph.to(dev), batch.features.to(dev))
                    probs = torch.softmax(logits, dim=1)

                    if loss_func is not None:
                        loss = loss_func(logits, label)
                        total_loss += loss.item()
                        num_batches += 1

                    _, preds = logits.max(1)

                    scores.append(probs.cpu().detach().numpy())
                    labels_all.append(label.cpu().detach().numpy())

                    correct = (preds == label).sum().item()
                    total_correct += correct
                    count += num_examples

                    postfix = {
                        'Acc': '%.5f' % (correct / num_examples),
                        'AvgAcc': '%.5f' % (total_correct / count),
                    }
                    if loss_func is not None and num_batches > 0:
                        postfix['AvgLoss'] = '%.5f' % (total_loss / num_batches)

                    tq.set_postfix(postfix)

        ts = time.time() - tic
        print('Tested over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(
            count=count, ts=ts, speed=count / ts
        ))

        avg_acc = total_correct / max(count, 1)
        avg_loss = (total_loss / max(num_batches, 1)) if loss_func is not None else None

        scores = np.concatenate(scores)
        labels_all = np.concatenate(labels_all)
        avg_auc = epoch_auc_from_scores(labels_all, scores[:, 1])

        if return_time:
            return ts
        if return_scores:
            return scores
        if return_metrics:
            return avg_loss, avg_acc, avg_auc
        return avg_acc

    # =========================
    # K-FOLD CROSS VALIDATION
    # =========================
    if cv_mode:
        assert(args.train_bkg and args.train_sig and
               args.val_bkg and args.val_sig and
               args.test_bkg and args.test_sig)

        if not (0.0 < args.cv_validation_fraction < 1.0):
            raise ValueError('--cv-validation-fraction should be comprise between 0 and 1.')

        if args.save and not os.path.exists(args.save):
            os.makedirs(args.save)

        data_train_all = DGLGraphDataset(args.train_bkg, args.train_sig, nev=args.nev, **dataset_kwargs)
        data_val_all = DGLGraphDataset(args.val_bkg, args.val_sig, nev=args.nev_val, **dataset_kwargs)
        data_test_all = DGLGraphDataset(args.test_bkg, args.test_sig, nev=args.nev_test, **dataset_kwargs)

        merged_data = ConcatDataset([data_train_all, data_val_all, data_test_all])
        merged_labels = np.concatenate([
            labels_to_numpy(data_train_all.label),
            labels_to_numpy(data_val_all.label),
            labels_to_numpy(data_test_all.label)
        ])

        input_dims = data_train_all.num_features

        print('Running %d-fold cross-validation on merged train+val+test sample...' % args.cv_folds)
        print('Merged dataset size:', len(merged_labels))
        print('Validation fraction inside non-test folds:', args.cv_validation_fraction)

        skf = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.cv_seed
        )

        cv_results = []

        for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(merged_labels)), merged_labels), start=1):
            print('\n' + '=' * 80)
            print('CV fold %d / %d' % (fold_idx, args.cv_folds))
            print('=' * 80)

            trainval_labels = merged_labels[trainval_idx]

            val_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=args.cv_validation_fraction,
                random_state=args.cv_seed + fold_idx
            )
            rel_train_idx, rel_val_idx = next(val_splitter.split(np.zeros(len(trainval_idx)), trainval_labels))

            train_idx = trainval_idx[rel_train_idx]
            val_idx = trainval_idx[rel_val_idx]

            fold_train_data = Subset(merged_data, train_idx.tolist())
            fold_val_data = Subset(merged_data, val_idx.tolist())
            fold_test_data = Subset(merged_data, test_idx.tolist())

            fold_train_loader = DataLoader(
                fold_train_data,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )
            fold_val_loader = DataLoader(
                fold_val_data,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )
            fold_test_loader = DataLoader(
                fold_test_data,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )

            model = Net(input_dims=input_dims, num_classes=2,
                        conv_params=conv_params,
                        fc_params=fc_params,
                        use_fusion=use_fusion)
            model = model.to(dev)

            num_total_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            num_buffers = sum(b.numel() for b in model.buffers())
            print("Total parameters:", num_total_params)
            print("Trainable parameters:", num_trainable_params)
            print("Buffers:", num_buffers)

            loss_func = torch.nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)
            lr_steps = [int(x) for x in args.lr_steps.split(',') if x.strip()]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)

            best_valid_acc = -1.0
            best_epoch = 0
            best_state_dict = copy.deepcopy(model.state_dict())

            history = {
                'epoch': [],
                'train_loss': [],
                'train_acc': [],
                'train_auc': [],
                'val_loss': [],
                'val_acc': [],
                'val_auc': [],
            }

            for epoch in range(args.num_epochs):
                print('Fold #%d Epoch #%d Training' % (fold_idx, epoch))
                train_loss, train_acc, train_auc = train(model, opt, scheduler, fold_train_loader, dev, loss_func)

                print('Fold #%d Epoch #%d Validating' % (fold_idx, epoch))
                valid_loss, valid_acc, valid_auc = evaluate(
                    model, fold_val_loader, dev,
                    loss_func=loss_func,
                    return_metrics=True
                )

                history['epoch'].append(epoch + 1)
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['train_auc'].append(train_auc)
                history['val_loss'].append(valid_loss)
                history['val_acc'].append(valid_acc)
                history['val_auc'].append(valid_auc)

                print(
                    'Fold %d Epoch %d summary | train_loss=%.5f train_acc=%.5f train_auc=%.5f | val_loss=%.5f val_acc=%.5f val_auc=%.5f'
                    % (fold_idx, epoch + 1, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc)
                )

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(model.state_dict())
                    if args.save:
                        torch.save(model.state_dict(), os.path.join(args.save, '%s_fold%d_state.pt' % (args.name, fold_idx)))

                if args.save:
                    with open(os.path.join(args.save, '%s_fold%d_history.pickle' % (args.name, fold_idx)), 'wb') as f:
                        pickle.dump(history, f)

                    with open(os.path.join(args.save, '%s_fold%d_history.csv' % (args.name, fold_idx)), 'w') as f:
                        f.write('epoch,train_loss,train_acc,train_auc,val_loss,val_acc,val_auc\n')
                        for e, tl, ta, tua, vl, va, vua in zip(
                            history['epoch'],
                            history['train_loss'],
                            history['train_acc'],
                            history['train_auc'],
                            history['val_loss'],
                            history['val_acc'],
                            history['val_auc'],
                        ):
                            f.write(f'{e},{tl},{ta},{tua},{vl},{va},{vua}\n')

                print('Current validation acc: %.5f (best: %.5f, epoch %d)' % (valid_acc, best_valid_acc, best_epoch))

            model.load_state_dict(best_state_dict)

            test_preds = evaluate(model, fold_test_loader, dev, return_scores=True)
            test_labels = merged_labels[test_idx]

            fpr, tpr, threshs = roc_curve(test_labels, test_preds[:, 1], pos_label=1)
            eff_s = tpr
            eff_b = 1 - fpr
            test_auc = ROC_area(eff_s, eff_b)
            test_acc = accuracy(test_preds, test_labels)

            fold_info = {
                'fold': fold_idx,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'n_test': len(test_idx),
            }
            cv_results.append(fold_info)

            print('Fold %d test_acc = %.6f' % (fold_idx, test_acc))
            print('Fold %d test_auc = %.6f' % (fold_idx, test_auc))

        path = args.save if args.save else '.'
        if path and not os.path.exists(path):
            os.makedirs(path)

        cv_csv = os.path.join(path, '%s_cv_results.csv' % args.name)
        with open(cv_csv, 'w') as f:
            f.write('fold,test_acc,test_auc,n_train,n_val,n_test\n')
            for row in cv_results:
                f.write('{fold},{test_acc},{test_auc},{n_train},{n_val},{n_test}\n'.format(**row))

        cv_pkl = os.path.join(path, '%s_cv_results.pickle' % args.name)
        with open(cv_pkl, 'wb') as f:
            pickle.dump(cv_results, f)

        mean_acc = np.mean([x['test_acc'] for x in cv_results])
        std_acc = np.std([x['test_acc'] for x in cv_results])
        mean_auc = np.mean([x['test_auc'] for x in cv_results])
        std_auc = np.std([x['test_auc'] for x in cv_results])

        cv_info = os.path.join(path, '%s_cv_summary.txt' % args.name)
        with open(cv_info, 'w') as f:
            f.write('cv_folds: %d\n' % args.cv_folds)
            f.write('cv_validation_fraction: %f\n' % args.cv_validation_fraction)
            f.write('mean_test_acc: %f\n' % mean_acc)
            f.write('std_test_acc: %f\n' % std_acc)
            f.write('mean_test_auc: %f\n' % mean_auc)
            f.write('std_test_auc: %f\n' % std_auc)
            f.write('\n')
            for row in cv_results:
                f.write('fold {fold}: test_acc={test_acc}, test_auc={test_auc}, n_train={n_train}, n_val={n_val}, n_test={n_test}\n'.format(**row))

        print('\n=== CV Summary ===')
        for row in cv_results:
            print('fold {fold}: test_acc={test_acc:.6f}, test_auc={test_auc:.6f}'.format(**row))
        print('mean_test_acc = %.6f +/- %.6f' % (mean_acc, std_acc))
        print('mean_test_auc = %.6f +/- %.6f' % (mean_auc, std_auc))

        return

    # =========================
    # STANDARD TRAINING MODE
    # =========================
    if training_mode:
        # loss function
        loss_func = torch.nn.CrossEntropyLoss()

        # optimizer
        opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)

        # learning rate
        lr_steps = [int(x) for x in args.lr_steps.split(',') if x.strip()]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)

        # training loop
        best_valid_acc = -1.0
        best_epoch = 0
        best_state_dict = copy.deepcopy(model.state_dict())

        history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_auc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
        }
        if args.save and not os.path.exists(args.save):
            os.makedirs(args.save)

        start_time = time.time()
        for epoch in range(args.num_epochs):
            print('Epoch #%d Training' % epoch)
            train_loss, train_acc, train_auc = train(model, opt, scheduler, train_loader, dev, loss_func)

            print('Epoch #%d Validating' % epoch)
            valid_loss, valid_acc, valid_auc = evaluate(
                model, val_loader, dev,
                loss_func=loss_func,
                return_metrics=True
            )

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_auc'].append(train_auc)
            history['val_loss'].append(valid_loss)
            history['val_acc'].append(valid_acc)
            history['val_auc'].append(valid_auc)

            print(
                'Epoch %d summary | train_loss=%.5f train_acc=%.5f train_auc=%.5f | val_loss=%.5f val_acc=%.5f val_auc=%.5f'
                % (epoch + 1, train_loss, train_acc, train_auc, valid_loss, valid_acc, valid_auc)
            )
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                if args.save:
                    torch.save(model.state_dict(), os.path.join(args.save, '%s_state.pt' % args.name))

            if args.save:
                # salvataggio history in pickle
                with open(os.path.join(args.save, '%s_history.pickle' % args.name), 'wb') as f:
                    pickle.dump(history, f)

                # salvataggio history in csv leggibile
                with open(os.path.join(args.save, '%s_history.csv' % args.name), 'w') as f:
                    f.write('epoch,train_loss,train_acc,train_auc,val_loss,val_acc,val_auc\n')
                    for e, tl, ta, tua, vl, va, vua in zip(
                        history['epoch'],
                        history['train_loss'],
                        history['train_acc'],
                        history['train_auc'],
                        history['val_loss'],
                        history['val_acc'],
                        history['val_auc'],
                    ):
                        f.write(f'{e},{tl},{ta},{tua},{vl},{va},{vua}\n')

            print('Current validation acc: %.5f (best: %.5f, epoch %d)' % (valid_acc, best_valid_acc, best_epoch))
        end_time = time.time()

    # =========================
    # STANDARD TEST EVALUATION
    # =========================
    path = args.save if training_mode else os.path.dirname(args.load)
    if not path:
        path = '.'
    name = args.test_output if args.test_output else 'test'
    if path and not os.path.exists(path):
        os.makedirs(path)

    if training_mode:
        del train_data, train_loader, val_data, val_loader
        test_data = DGLGraphDataset(args.test_bkg, args.test_sig, nev=args.nev_test, **dataset_kwargs)
        test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)

    test_labels = labels_to_numpy(test_data.label)
    test_preds = np.zeros((len(test_labels), 2), dtype='float32')

    # load saved model
    if training_mode:
        model.load_state_dict(best_state_dict)
    else:
        model_path = args.load
        if not model_path.endswith('.pt'):
            model_path = os.path.join(model_path, '%s_state.pt' % args.name)
        print('Loading model %s for eval' % model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))

    test_preds += evaluate(model, test_loader, dev, return_scores=True)

    info_dict = {'model_name': args.model,
                 'model_params': {'conv_params': conv_params, 'fc_params': fc_params},
                 'lund_ln_kt_min': args.ln_kt_min,
                 'lund_ln_delta_min': args.ln_delta_min,
                 'max_depth': args.max_depth if 'lund' in args.model else 'N/A',
                 'date': str(datetime.date.today()),
                 'model_path': args.save if training_mode else args.load,
                 'model_name': args.name,
                 'test_sig': args.test_sig,
                 'test_bkg': args.test_bkg}
    if training_mode:
        info_dict.update({'train_sig': args.train_sig,
                          'train_bkg': args.train_bkg,
                          'training_time': str(end_time - start_time) + " seconds"}
                        )

    base_name = name.split('.')[0]

    test_pred_labels = np.argmax(test_preds, axis=1)

    with open(os.path.join(path, base_name + '_predictions.pickle'), 'wb') as f:
        pickle.dump({
            'true_labels': test_labels,
            'pred_labels': test_pred_labels,
            'pred_probs': test_preds,
            'prob_bkg': test_preds[:, 0],
            'prob_sig': test_preds[:, 1],
        }, f)

    with open(os.path.join(path, base_name + '_predictions.csv'), 'w') as f:
        f.write('index,true_label,pred_label,prob_bkg,prob_sig\n')
        for i in range(len(test_labels)):
            f.write('{},{},{},{},{}\n'.format(
                i,
                int(test_labels[i]),
                int(test_pred_labels[i]),
                float(test_preds[i, 0]),
                float(test_preds[i, 1])
            ))

    fpr, tpr, threshs = roc_curve(test_labels, test_preds[:, 1], pos_label=1)
    # convert into signal and background efficiency
    eff_s = tpr
    eff_b = 1 - fpr
    auc = ROC_area(eff_s, eff_b)

    info_dict['best_epoch'] = best_epoch
    info_dict['best_valid_acc'] = best_valid_acc
    info_dict['accuracy'] = accuracy(test_preds, test_labels)
    info_dict['auc'] = auc
    info_dict['inv_bkg_at_sig_50'] = bkg_rejection_at_threshold(eff_s, eff_b, 0.5)
    info_dict['inv_bkg_at_sig_30'] = bkg_rejection_at_threshold(eff_s, eff_b, 0.3)

    print(' === Summary ===')
    for k in info_dict:
        print('%s: %s' % (k, info_dict[k]))

    info_file = os.path.join(path, args.name if training_mode else name) + '_INFO.txt'
    with open(info_file, 'w') as f:
        for k in info_dict:
            f.write('%s: %s\n' % (k, info_dict[k]))

    filename = os.path.join(path, base_name)
    with open(filename + '_ROC_data.pickle', 'wb') as f:
        pickle.dump({'signal_eff': eff_s,
                     'background_eff': eff_b,
                     'thresholds': threshs,
                     'description': str(args)}, f)

    print('Saving ROC data for %s' % base_name)