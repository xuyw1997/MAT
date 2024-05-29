import torch
import torch.optim
import numpy as np
import time
import utils.distributed as du
import utils.checkpoint as cu
import utils.misc as misc
from model.build import build_model
import model.optimizer as optim
from dataset import loader
from utils.meters import TrainMeter, ValMeter, EpochTimer
import visualization.tensorboard_vis as tb
import model.losses as losses
from utils import logging
import pprint
from fvcore.nn import FlopCountAnalysis, parameter_count

logger = logging.get_logger(__name__)


def train_epoch3(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    
    misc.frozen_bn_stats(model)

    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.LOSS.NAME)(cfg)
    
    for cur_iter, (frames, word_vectors, text_masks, word_label, word_mask, label, times, duration, clip_mask) in enumerate(train_loader):
        batch_size = frames.shape[0]
        num_clip_per_video = cfg.DATA.NUM_SAMPLE_FRAME // cfg.DATA.WINDOW_SIZE
        clip_step = cur_iter % num_clip_per_video

            
        # Transfer the data to the current GPU device.
        if cfg.DDP.NUM_GPUS:
            word_vectors = word_vectors.cuda()
            text_masks = text_masks.cuda()
            label = label.cuda()
            word_label = word_label.cuda()
            word_mask = word_mask.cuda()

        # frames = frames.view((-1, cfg.DATA.WINDOW_SIZE, 3) +  frames.size()[-2:])
        
        # Update the learning rate.
        if clip_step == 0:
            epoch_exact = cur_epoch + float(cur_iter) / data_size
            lr = optim.get_epoch_lr(epoch_exact, cfg)
            optim.set_lr(optimizer, lr)

        if  clip_step == 0:
            mems = None
            c_mems = None

        clip = frames
        if cfg.DDP.NUM_GPUS:
            clip = clip.cuda()
        train_meter.data_toc()
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            loc, conf, centerness, clip_level_pred, mems, c_mems= model(clip, word_vectors, text_masks, word_mask, mems=mems, c_mems=c_mems)
            loss = loss_fun(loc, conf, centerness, clip_level_pred,label, clip_step * cfg.DATA.WINDOW_SIZE)
            

        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        if ((clip_step + 1) % cfg.SOLVER.GRAD_ACC_STEP == 0) or ((clip_step + 1) == num_clip_per_video):
            scaler.unscale_(optimizer)
            # Clip gradients if necessary
            if cfg.SOLVER.CLIP_GRAD_VAL:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
                )
            elif cfg.SOLVER.CLIP_GRAD_L2NORM:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
                )

             # Update the parameters.
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update() 
            
        # Update and log stats.
        train_meter.update_stats(
            loss.item(),
            lr,
            batch_size
            * max(
                cfg.DDP.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.DDP.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        # write to tensorboard format if available.
        if writer is not None and (cur_iter + 1) % cfg.LOG_PERIOD == 0:
            writer.add_scalars(
                {
                    "Train/loss": loss.item(),
                    "Train/lr": lr,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )

        # torch.cuda.synchronize()
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

def report(model, *inputs):
    x = FlopCountAnalysis(model, inputs)
    # print(flop_count_table(x))
    print("flops: ", x.total())
    print("params: ", parameter_count(model)[""])
    return x.total()
    # macs, params = profile(model, inputs=inputs)
    # return macs
    
    

@torch.no_grad()
def eval_epoch3(
    val_loader, model, val_meter, cur_epoch, cfg, writer, save_preds=False
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    loss_fun = losses.get_loss_func(cfg.LOSS.NAME)(cfg)
    pure_inf_time = 0
    for cur_iter, (frames, word_vectors, text_masks, word_label, word_mask, label, times, duration, clip_mask) in enumerate(val_loader):
        batch_size = frames.shape[0]
        num_clip_per_video = cfg.DATA.NUM_SAMPLE_FRAME // cfg.DATA.WINDOW_SIZE
        clip_step = cur_iter % num_clip_per_video

        if cfg.DDP.NUM_GPUS:
            word_vectors = word_vectors.cuda()
            text_masks = text_masks.cuda()
            label = label.cuda()
            word_mask = word_mask.cuda()
            word_label = word_label.cuda()


        if  clip_step == 0:
            mems = None
            c_mems = None
        clip = frames
        if cfg.DDP.NUM_GPUS:
            clip = clip.cuda()
        val_meter.data_toc()
        if clip_step == 0:
            locs = []
            confs = []
            centers = []
            clip_levels = []
            total_flops = 0

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            loc, conf, centerness, clip_level_pred, mems, c_mems= model(clip, word_vectors, text_masks, word_mask, mems=mems, c_mems=c_mems)

            loss = loss_fun(loc, conf, centerness, clip_level_pred,label, clip_step * cfg.DATA.WINDOW_SIZE)
        

        locs.append(loc)
        confs.append(conf)
        centers.append(centerness)

        clip_levels.append(clip_level_pred.expand(-1, cfg.DATA.WINDOW_SIZE))
        
        if clip_step + 1 == num_clip_per_video:

            locs = torch.cat(locs, dim=1)
            confs = torch.cat(confs, dim=1)
            centers = torch.cat(centers, dim=1)
            clip_levels = torch.cat(clip_levels, dim=1)
            
            val_meter.update_stats7(loss.item(), locs, confs, centers, clip_levels, loss_fun.center, times, duration, clip_mask)
        
        
        val_meter.iter_toc()
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
            writer.add_scalars(
                {"Val/Iou=0.5": val_meter.r_accumulate[0] / val_meter.num_samples,
                 "Val/Iou=0.7": val_meter.r_accumulate[1] / val_meter.num_samples,
                 "Val/mIoU": sum(val_meter.all_ious) / val_meter.num_samples,
                 "val/loss": val_meter.loss_total / val_meter.num_loss_samples
                 }, global_step=cur_epoch
            )
    if save_preds:
        val_meter.save_preds()
    val_meter.reset()

    
def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.CONFIG_FILE)


    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)


    if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.DDP.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "test")

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.DDP.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    print("hello")
    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch3(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            writer,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = (cu.is_checkpoint_epoch(cfg,cur_epoch,) and cfg.TRAIN.SAVE_CHECKPOINT)
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch,
        )

        # Save a checkpoint.
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
                comment=time_str
            )
           
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch3(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                writer
            )
    if writer is not None:
        writer.close()
    logger.info("training done")



