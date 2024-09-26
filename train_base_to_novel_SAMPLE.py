import os
import os.path as osp
from collections import OrderedDict
import math, copy

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import time


import matplotlib.pyplot as plt
import numpy as np
import math
from utils.KLLoss import *
from test_SAMPLE import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import *

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


#!New Added#####################################################################


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.mm_prompt.N_CTX
        ctx_init = cfg.mm_prompt.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.data.input_size
        # Default is 1, which is compound shallow prompting
        assert cfg.mm_prompt.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.mm_prompt.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of mm_prompt context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        '''
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        '''

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, label=None):
        if label is not None:
            prefix = self.token_prefix[label]
            suffix = self.token_suffix[label]
        else:
            prefix = self.token_prefix
            suffix = self.token_suffix
            
        
        ctx = self.ctx
        if label is not None:
            ctx = ctx.unsqueeze(0).expand(prefix.shape[0], -1, -1)
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, prefix, suffix, label)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_features, text_features, label=None):
        logit_scale = self.logit_scale.exp()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        print(f'logits.shape:{logits.shape}')
        print(f'label.shape:{label.shape}')
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits
        
    
    def encode(self, image, label=None):
        if label is not None:
            tokenized_prompts = self.tokenized_prompts[label.cpu()]
        else:
            tokenized_prompts = self.tokenized_prompts
            
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(label)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        return  image_features, text_features
    
#!##############################################################################


def print_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def generate_triplet_samples(image_embeddings, text_embeddings, list_ids):
    anchor_indices = []
    positive_indices = []
    negative_indices = []

    for idx, anchor_class in enumerate(list_ids):
        # Find positive and negative indices
        positive_indices_for_class = [
            i for i, id in enumerate(list_ids) if id == anchor_class
        ]
        negative_indices_for_class = [
            i for i, id in enumerate(list_ids) if id != anchor_class
        ]

        positive_idx = random.choice(positive_indices_for_class)
        negative_idx = random.choice(negative_indices_for_class)

        anchor_indices.append(idx)
        positive_indices.append(positive_idx)
        negative_indices.append(negative_idx)

    anchor_images = image_embeddings[anchor_indices]
    anchor_texts = text_embeddings[anchor_indices]
    positive_images = image_embeddings[positive_indices]
    positive_texts = text_embeddings[positive_indices]
    negative_images = image_embeddings[negative_indices]
    negative_texts = text_embeddings[negative_indices]

    return {
        "anchor_image": anchor_images,
        "anchor_text": anchor_texts,
        "positive_image": positive_images,
        "positive_text": positive_texts,
        "negative_image": negative_images,
        "negative_text": negative_texts,
    }


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", default="")
    parser.add_argument("--traning_name", default="")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    args.traning_name = config["training_name"]
    working_dir = os.path.join(
        config["weight_save_dir"],
        config["network"]["type"],
        config["network"]["arch"],
        config["data"]["dataset"],
        args.traning_name,
    )
    print("-" * 80)
    print(" " * 20, "working dir: {}".format(working_dir))
    print("-" * 80)

    print("-" * 80)
    print(" " * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print("-" * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # If using GPU then use mixed precision training.

    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": config.mm_prompt.N_CTX,
    }
    model, clip_state_dict = clip.load(
        config.network.arch,
        config,
        device=torch.device("cpu"),
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint=config.network.joint,
        design_details=design_details,
    )  # Must set jit=False for training  ViT-B/32
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    print("train transforms: {}".format(transform_train.transforms))
    print("val transforms: {}".format(transform_val.transforms))
    ############################## base dataset  loader ###################################
    train_data = Action_DATASETS(
        config.data.train_list,
        config.data.label_list,
        num_segments=config.data.num_segments,
        image_tmpl=config.data.image_tmpl,
        random_shift=config.data.random_shift,
        transform=transform_train,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    val_data = Action_DATASETS(
        config.data.val_list,
        config.data.label_list,
        random_shift=False,
        num_segments=config.data.num_segments,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )
    ###################################################################################################################################################

    ############################## novel loader ###################################
    novel_val_data = Action_DATASETS(
        config.data.novel_val_list,
        config.data.novel_label_list,
        random_shift=False,
        num_segments=config.data.num_segments,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val,
    )
    novel_val_loader = DataLoader(
        novel_val_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )
    ###################################################################################################################################################

    #! Added#######
    classnames = [name for id, name in train_data.classes]
    customCLIP = CustomCLIP(config, classnames, model).to(device)
    print("Turning off gradients in both the image and the text encoder")
    for name, param in customCLIP.named_parameters():
        if (
            "prompt_learner" not in name
            and "prompt" not in name
            and "Adapter" not in name
        ):  # EZ_CLIP + CoOp
            # if "prompt_learner" not in name : # only CoOp
            # if "prompt" not in name and "Adapter" not in name or "prompt_learner" in name: # EZ-CLIP
            param.requires_grad_(False)
    customCLIP = torch.nn.DataParallel(customCLIP, device_ids=[0]).cuda()
    #! ############
    """ #! original
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    """
    text_param_count = 0
    param_count_with_prompts = 0
    param_count_with_adapters = 0
    visual_param_count_with_adapters = 0
    visual_param_count_with_T_adapters = 0
    for name, param in customCLIP.named_parameters():
        if "prompt_learner" in name:
            print("prompt_learner--", name)
            text_param_count += param.numel()
        elif "prompt" in name:
            param_count_with_prompts += param.numel()
        if "Adapter" in name and not "visual" in name:
            print("text---", name)
            param_count_with_adapters += param.numel()
        if "Adapter" in name and "visual" in name:
            print("visual--", name)
            visual_param_count_with_adapters += param.numel()
        if "T_Adapter" in name and "visual" in name:
            print("temporal visual--", name)
            visual_param_count_with_T_adapters += param.numel()

    param_count_with_prompts_in_million = param_count_with_prompts / 1_000_000
    param_count_with_adapter_in_million = (param_count_with_adapters) / 1_000_000
    T_visual_param_count_with_adapter_in_million = (
        visual_param_count_with_T_adapters / 1_000_000
    )
    S_visual_param_count_with_adapter_in_million = (
        visual_param_count_with_adapters - visual_param_count_with_T_adapters
    ) / 1_000_000
    text_param_count_in_million = text_param_count / 1_000_000
    # Print the count
    print(
        f'Number of Trainable Parameters with "prompts" in their names: {param_count_with_prompts_in_million:.3f}'
    )
    print(
        f'Number of Trainable Parameters with "Text adapters" in their names: {param_count_with_adapter_in_million:.3f}'
    )
    print(
        f'Number of Trainable Parameters with "S visual adapters" in their names: {S_visual_param_count_with_adapter_in_million:.3f}'
    )
    print(
        f'Number of Trainable Parameters with "T visual adapters" in their names: {T_visual_param_count_with_adapter_in_million:.3f}'
    )
    print(
        f'Number of Trainable Parameters with "prompt_learner" in their names: {text_param_count_in_million:.3f}'
    )

    """
    for name, p in model.named_parameters():
        if "prompt" not in name and "Adapter" not in name:
            p.requires_grad = False
    """
    ###########################################################
    parameters = filter(lambda p: p.requires_grad, customCLIP.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Modified CLIP_model Trainable Parameters: %.3fM" % parameters)

    ##########################################################
    """
    loss_img = KLLoss()
    loss_txt = KLLoss()
    """
    loss_img = nn.CrossEntropyLoss()
    loss_motion = Motion_loss()

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            state_dict = checkpoint["model_state_dict"]
            # Ignore the fixed token vectors
            if "module.prompt_learner.token_prefix" in state_dict:
                del state_dict["module.prompt_learner.token_prefix"]
            if "module.prompt_learner.token_suffix" in state_dict:
                del state_dict["module.prompt_learner.token_suffix"]
            customCLIP.load_state_dict(state_dict, strict=False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            state_dict = checkpoint["model_state_dict"]
            # Ignore the fixed token vectors
            if "module.prompt_learner.token_prefix" in state_dict:
                del state_dict["module.prompt_learner.token_prefix"]
            if "module.prompt_learner.token_suffix" in state_dict:
                del state_dict["module.prompt_learner.token_suffix"]
            customCLIP.load_state_dict(state_dict, strict=False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    ##### Novel########
    novel_classes, novel_num_text_aug, novel_text_dict = text_prompt(novel_val_data,config.data.novel_gpt_discription, config.data.use_llm)
    optimizer = _optimizer(config, model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    loss=[]
    top_1_acc=[]
    top_5_acc=[]
    novel_top_1_acc=[]    
    novel_top_5_acc=[]
    best_prec1 = 0.0
    novel_best_prec1 = 0.0

    if config.solver.evaluate:
        prec1, prec5 = validate(
            start_epoch,
            val_loader,
            # classes,
            device,
            customCLIP,
            config,
            # num_text_aug,
            working_dir,
            f'base_{config["data"]["dataset"]}',
            labels2name,
            is_Train=False,
        )
        novel_prec1, novel_prec5 = validate(
            start_epoch,
            novel_val_loader,
            # classes,
            device,
            customCLIP,
            config,
            # num_text_aug,
            working_dir,
            f'novel_{config["data"]["dataset"]}',
            labels2name,
            is_Train=False,
        )
        print("{} Base Testing: {}/{}".format(config.data.dataset, prec1, best_prec1))
        print("{} Novel Testing: {}/{}".format(config.data.dataset, novel_prec1, novel_best_prec1))
        return

    for k, v in model.named_parameters():
        if v.requires_grad:
            print("{}: {}".format(k, v.requires_grad))

    for epoch in range(start_epoch, config.solver.epochs):
        print(
            "------------------------------------------------------------------------"
        )
        print("Epoch %d start .." % epoch)
        """
        model_image.train()
        model_text.train()
        """
        customCLIP.train()
        tic = time.time()
        epoch_loss = []
        for kkk, (prompt_images, list_id) in enumerate((train_loader)):
            prompt_images = prompt_images.to(device)
            list_id = list_id.to(device)
            if config.solver.type != "monitor":
                if (kkk + 1) == 1 or (kkk + 1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()
            # prompt_images = prompter(images)

            prompt_images = prompt_images.view(
                (-1, config.data.num_segments, 3) + prompt_images.size()[-2:]
            )
            b, t, c, h, w = prompt_images.size()
            """
            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i, :] for i, j in zip(list_id, text_id)])
            """
            prompt_images = prompt_images.view(
                -1, c, h, w
            )  # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            # texts = texts.to(device)
            image_embedding, text_features = customCLIP.module.encode(
                prompt_images
            )

            image_embedding = image_embedding.view(b, t, -1)
            if config.use_motion_loss:
                loss_video_motion = loss_motion(image_embedding)
            image_embedding = image_embedding.mean(dim=1, keepdim=False)

            """
            text_embedding = customCLIP.module.encode_text()

            if config.network.fix_text:
                text_embedding.detach_()
            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(
                image_embedding, text_embedding, logit_scale
            )
            """
            loss_imgs = customCLIP(image_embedding, text_features, list_id)
            """
            ground_truth = torch.tensor(
                gen_label(list_id), dtype=image_embedding.dtype, device=device
            )
            """
            #!To Remove##############################
            """
            print(f'image_embedding.shape:{image_embedding.shape}, text_embedding.shape:{f.shape}')
            print(f'logits_per_image.shape:{logits_per_image.shape}, ground_truth.shape:{ground_truth.shape}')
            print(f'logits_per_text.shape:{logits_per_text.shape}, ground_truth.shape:{ground_truth.shape}')
            image_embedding.shape:torch.Size([16, 512]), text_embedding.shape:torch.Size([58, 512])
            logits_per_image.shape:torch.Size([16, 58]), ground_truth.shape:torch.Size([16, 16])
            logits_per_text.shape:torch.Size([58, 16]), ground_truth.shape:torch.Size([16, 16])
            """
            #!##############################
            # loss_imgs = loss_img(logits_per_image, ground_truth)
            # loss_texts = loss_txt(logits_per_text, ground_truth)
            # list_id = torch.tensor(list_id).long().to(device=device)
            # loss_imgs = loss_img(logits_per_image, list_id)
            """
            if config.use_motion_loss:
                total_loss = (loss_imgs + loss_texts) / 2 + loss_video_motion
            else:
                total_loss = (loss_imgs + loss_texts) / 2
            """
            if config.use_motion_loss:
                total_loss = loss_imgs + loss_video_motion
            else:
                total_loss = loss_imgs

            epoch_loss.append(total_loss.item())
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            if kkk % 100 == 0:
                if config.use_motion_loss:
                    print(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, motion loss:%f, lr:%f "
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss_imgs.item(),
                            # loss_texts.item(),
                            loss_video_motion.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    print(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, lr:%f "
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss_imgs.item(),
                            # loss_texts.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            print("{} val accuracy".format(config.data.dataset))
            novel_prec1, novel_prec5 = validate(
                epoch,
                novel_val_loader,
                # classes,
                device,
                customCLIP,
                config,
                # num_text_aug,
                working_dir,
                f'novel_{config.data.dataset}',
                is_Train=True,
            )
            prec1, prec5 = validate(
                epoch,
                val_loader,
                # classes,
                device,
                customCLIP,
                config,
                # num_text_aug,
                working_dir,
                f'base_{config.data.dataset}',
                is_Train=True,
            )
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print("{} Base Testing: {}/{}".format(config.data.dataset, prec1, best_prec1))
        novel_is_best = novel_prec1 > novel_best_prec1
        novel_best_prec1 = max(novel_prec1, novel_best_prec1)
        print("{} Novel Testing: {}/{}".format(config.data.dataset, novel_prec1, novel_best_prec1))

        txt_path = "{}/log.txt".format(working_dir)
        if os.path.exists(txt_path):
            with open(txt_path, "a+") as f:
                f.write("\n")
                if config.use_motion_loss:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, motion loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss_imgs.item(),
                            # loss_texts.item(),
                            loss_video_motion.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss_imgs.item(),
                            # loss_texts.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                f.write(
                    "{} Base Testing: top1:{}/{}, top5:{}\n".format(config.data.dataset, prec1, best_prec1, prec5)
                )
                f.write(
                    "{} Novel Testing: top1:{}/{}, top5:{}\n".format(config.data.dataset, novel_prec1, novel_best_prec1, novel_prec5)
                )
                f.close()
        else:
            with open(txt_path, mode="wt") as f:
                if config.use_motion_loss:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, motion loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss_imgs.item(),
                            # loss_texts.item(),
                            loss_video_motion.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, image loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss_imgs.item(),
                            # loss_texts.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                f.write(
                    "{} Base Testing: top1:{}/{}, top5:{}\n".format(config.data.dataset, prec1, best_prec1, prec5)
                )
                f.write(
                    "{} Novel Testing: top1:{}/{}, top5:{}\n".format(config.data.dataset, novel_prec1, novel_best_prec1, novel_prec5)
                )
                f.close()

        print("Saving:")
        filename1 = "{}/last_model.pt".format(working_dir)
        # filename = "{}/epoch_{}_model.pt".format(working_dir,epoch)
        top_1_acc.append(prec1 / 100)
        top_5_acc.append(prec5 / 100)
        novel_top_1_acc.append(novel_prec1 / 100)
        novel_top_5_acc.append(novel_prec5 / 100)
        loss.append(np.mean(epoch_loss))
        # epoch_saving(epoch, model,  optimizer, filename)
        epoch_saving(epoch, customCLIP, optimizer, filename1)
        if is_best:
            print(
                "Saving best weight based on {} Base accuracy at epoch {}".format(
                    config.data.dataset, epoch
                )
            )
            best_saving(working_dir, epoch, customCLIP, optimizer, f'base_{config.data.dataset}')
        if novel_is_best:
            print(
                "Saving best weight based on {} Novel accuracy at epoch {}".format(
                    config.data.dataset, epoch
                )
            )
            best_saving(working_dir, epoch, customCLIP, optimizer, f'novel_{config.data.dataset}')

        print("Epoch %d end .." % epoch)
        ##############graph_plot################
        X = list(range(len(loss)))
        plt.plot(X, loss, color="r", label="Training loss")
        plt.plot(
            X, top_1_acc, color="g", label="{} Base Accuracy".format(config.data.dataset)
        )
        plt.plot(
            X, novel_top_1_acc, color="g", label="{} Novel Accuracy".format(config.data.dataset)
        )

        plt.xlabel("Epoch")
        plt.ylabel("Training loss and Accuracy")
        plt.title("Traing graph")
        plt.legend()
        plt.savefig("{}/Graph_plot.png".format(working_dir))
        plt.close()
        print("Time taken by epoch %d:" % epoch, print_time(time.time() - tic))
        print(
            "------------------------------------------------------------------------"
        )
    print("====================Final Testing:=================")
    # labels2name: dict {'0': 'xxx',}
    labels_csv_path = config.data.label_list
    with open(labels_csv_path, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        labels2name = {int(row[0]): row[1] for row in reader}
    print(labels2name)
    prec1, prec5 = validate(
        start_epoch,
        val_loader,
        # classes,
        device,
        customCLIP,
        config,
        # num_text_aug,
        working_dir,
        f'base_{config["data"]["dataset"]}',
        labels2name,
        is_Train=False,
    )
    novel_prec1, novel_prec5 = validate(
        start_epoch,
        novel_val_loader,
        # classes,
        device,
        customCLIP,
        config,
        # num_text_aug,
        working_dir,
        f'novel_{config["data"]["dataset"]}',
        labels2name,
        is_Train=False,
    )
    print("{} Base Testing: {}/{}".format(config.data.dataset, prec1, best_prec1))
    print("{} Novel Testing: {}/{}".format(config.data.dataset, novel_prec1, novel_best_prec1))
    # log the results into txt_path
    if os.path.exists(txt_path):
        with open(txt_path, "a+") as f:
            f.write("{} Base Testing: {}/{}\n".format(config.data.dataset, prec1, best_prec1))
            f.write("{} Novel Testing: {}/{}\n".format(config.data.dataset, novel_prec1, novel_best_prec1))
            f.close()
    else:
        with open(txt_path, "w") as f:
            f.write("{} Base Testing: {}/{}\n".format(config.data.dataset, prec1, best_prec1))
            f.write("{} Novel Testing: {}/{}\n".format(config.data.dataset, novel_prec1, novel_best_prec1))
            f.close()
    print("Results logged into {}".format(txt_path))


if __name__ == "__main__":
    main()
