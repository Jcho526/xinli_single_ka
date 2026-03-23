import argparse
import json
import math
import os
import sys
import logging
import traceback
import faulthandler

import torch
from torch.utils.data import DataLoader, RandomSampler
import deepspeed
from tqdm import tqdm

from utils import (
    print_trainable_parameters,
    to_device,
    set_random_seed,
    save_model,
    DataCollator
)
from peft import LoraConfig, get_peft_model
from model import MODE

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def setup_realtime_logging():
    # 开启 faulthandler，崩溃时能立刻把栈打到前台
    faulthandler.enable()

    # 尽量强制 stdout/stderr 行缓冲，保证实时输出
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    os.environ["PYTHONUNBUFFERED"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def log_rank_0(msg, rank=0, level=logging.INFO):
    if rank <= 0:
        logging.log(level, msg)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass


def install_excepthook():
    def _hook(exc_type, exc_value, exc_tb):
        logging.error("Uncaught exception:")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

    sys.excepthook = _hook


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    # DataSet
    parser.add_argument("--train_path", default="", type=str)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--is_skip", action="store_true")
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--mode", type=str, default="glm2")
    parser.add_argument("--train_type", type=str, default="lora")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--show_loss_step", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_model_step", type=int, default=500)
    # deepspeed
    parser.add_argument("--ds_file", type=str, default="ds_zero2.json")
    # LoRA
    parser.add_argument("--lora_dim", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=30)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_module_name", type=str, default="query_key_value")
    # Freeze
    parser.add_argument("--freeze_module_name", type=str, default="layers.27.")
    # P-tuning
    parser.add_argument("--pre_seq_len", type=int, default=16)
    parser.add_argument("--prefix_projection", type=bool, default=True)

    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    setup_realtime_logging()
    install_excepthook()

    args = parse_args()


    if args.local_rank == -1:
        args.local_rank = 0
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    args.global_rank = 0

    log_rank_0(f"Using device: {device}", args.global_rank)
    log_rank_0(f"model_name_or_path = {args.model_name_or_path}", args.global_rank)
    log_rank_0(f"train_path = {args.train_path}", args.global_rank)
    log_rank_0(f"ds_file = {args.ds_file}", args.global_rank)
    log_rank_0(f"mode = {args.mode}, train_type = {args.train_type}", args.global_rank)

    # load deepspeed config
    log_rank_0("Loading DeepSpeed config...", args.global_rank)
    with open(args.ds_file, "r", encoding="utf-8") as fh:
        ds_config = json.load(fh)

    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    tb_write = None
    if args.global_rank <= 0:
        tb_write = SummaryWriter()

    set_random_seed(args.seed)
    log_rank_0(f"Seed set to {args.seed}", args.global_rank)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # load tokenizer
    log_rank_0("Loading tokenizer...", args.global_rank)
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.model_name_or_path)
    log_rank_0("Tokenizer loaded.", args.global_rank)

    # load model
    log_rank_0("Loading model...", args.global_rank)
    if args.train_type == "lora":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
        lora_module_name = args.lora_module_name.split(",")
        config = LoraConfig(
            r=args.lora_dim,
            lora_alpha=args.lora_alpha,
            target_modules=lora_module_name,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        model = get_peft_model(model, config)
        model.config.torch_dtype = torch.float32

    elif args.train_type == "freeze":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
        freeze_module_name = args.freeze_module_name.split(",")
        for name, param in model.named_parameters():
            if not any(nd in name for nd in freeze_module_name):
                param.requires_grad = False

    elif args.train_type == "ptuning":
        config = MODE[args.mode]["config"].from_pretrained(args.model_name_or_path)
        config.pre_seq_len = args.pre_seq_len
        config.prefix_projection = args.prefix_projection
        model = MODE[args.mode]["model"].from_pretrained(
            args.model_name_or_path, config=config
        )
        for name, param in model.named_parameters():
            if not any(nd in name for nd in ["prefix_encoder"]):
                param.requires_grad = False

    elif args.train_type == "all":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
    else:
        raise Exception("train_type 无效")

    log_rank_0("Model loaded.", args.global_rank)

    # load data
    log_rank_0("Loading dataset...", args.global_rank)
    train_dataset = MODE[args.mode]["dataset"](
        args.train_path, tokenizer, args.max_len, args.max_src_len, args.is_skip
    )
    log_rank_0(f"Dataset loaded. size = {len(train_dataset)}", args.global_rank)

    train_sampler = RandomSampler(train_dataset)

    data_collator = DataCollator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )
    log_rank_0(f"DataLoader ready. steps/epoch = {len(train_dataloader)}", args.global_rank)

    # optimizer config
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
    ds_config["optimizer"]["params"]["betas"] = (0.9, 0.95)
    ds_config["optimizer"]["params"]["eps"] = 1e-8
    ds_config["optimizer"]["params"]["weight_decay"] = args.weight_decay

    num_training_steps = args.num_train_epochs * math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_min_lr"] = args.learning_rate * 0.1

    print_trainable_parameters(model)

    if args.gradient_checkpointing:
        log_rank_0("Enabling gradient checkpointing...", args.global_rank)
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    log_rank_0("Initializing DeepSpeed...", args.global_rank)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        args=args,
        config=ds_config,
        dist_init_required=True,
    )
    log_rank_0("DeepSpeed initialized.", args.global_rank)

    model.train()
    tr_loss, logging_loss = 0.0, 0.0
    global_step = 0

    for epoch in range(args.num_train_epochs):
        log_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, "
            f"Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            unit="batch",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )

        for step, batch in progress_bar:
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            tr_loss += loss.item()

            model.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.step()

            progress_bar.set_postfix(
                epoch=epoch + 1,
                step=step + 1,
                loss=f"{loss.item():.4f}"
            )

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1

                if global_step % args.show_loss_step == 0:
                    avg_loss = (tr_loss - logging_loss) / (
                        args.show_loss_step * args.gradient_accumulation_steps
                    )
                    log_rank_0(
                        f"Epoch: {epoch}, step: {step + 1}, "
                        f"global_step: {global_step}, loss: {avg_loss}",
                        args.global_rank,
                    )
                    if args.global_rank <= 0 and tb_write is not None:
                        tb_write.add_scalar("train_loss", avg_loss, global_step)
                        logging_loss = tr_loss

                if args.save_model_step and global_step % args.save_model_step == 0:
                    log_rank_0(
                        f"Saving checkpoint at epoch={epoch + 1}, global_step={global_step}",
                        args.global_rank,
                    )
                    if ds_config["zero_optimization"]["stage"] == 3:
                        state_dict = model._zero3_consolidated_16bit_state_dict()
                        if args.global_rank <= 0:
                            save_model(
                                model,
                                tokenizer,
                                args.output_dir,
                                f"epoch-{epoch + 1}-step-{global_step}",
                                state_dict,
                            )
                    else:
                        if args.global_rank <= 0:
                            save_model(
                                model,
                                tokenizer,
                                args.output_dir,
                                f"epoch-{epoch + 1}-step-{global_step}",
                            )

    log_rank_0("Training finished. Saving final model...", args.global_rank)

    if ds_config["zero_optimization"]["stage"] == 3:
        state_dict = model._zero3_consolidated_16bit_state_dict()
        if args.global_rank <= 0:
            save_model(
                model,
                tokenizer,
                args.output_dir,
                "final",
                state_dict,
            )
    else:
        if args.global_rank <= 0:
            save_model(model, tokenizer, args.output_dir, "final")

    log_rank_0("Final model saved.", args.global_rank)

    if tb_write is not None:
        tb_write.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.error("Training failed with an exception:")
        traceback.print_exc()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        raise