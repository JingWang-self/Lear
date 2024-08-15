import os
import os.path as osp
from collections import OrderedDict
import math
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
from imagenet_templates import IMAGENET_TEMPLATES


import matplotlib.pyplot as plt
import numpy as np
import math
from utils.KLLoss import *
from test_promptSRC import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import *

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()



#!New Added#####################################################################
def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.network.arch
    if backbone_name in clip._MODELS:
        model_path = clip._download(clip._MODELS[backbone_name])
    elif os.path.isfile(backbone_name):
        model_path = backbone_name
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.PROMPTSRC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.PROMPTSRC.N_CTX_VISION,
                          "language_ctx": cfg.PROMPTSRC.N_CTX_TEXT}
        model, clip_state_dict = clip.load(
            cfg.network.arch,
            cfg,
            device=torch.device('cpu'),
            jit=False,
            tsm=cfg.network.tsm,
            T=cfg.data.num_segments,
            dropout=cfg.network.drop_out,
            emb_dropout=cfg.network.emb_dropout,
            pretrain=cfg.network.init,
            joint=cfg.network.joint,
            design_details=design_details
        )  # Must set jit=False for training  ViT-B/32
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model, clip_state_dict = clip.load(
            cfg.network.arch,
            cfg,
            device=torch.device('cpu'),
            jit=False,
            tsm=cfg.network.tsm,
            T=cfg.data.num_segments,
            dropout=cfg.network.drop_out,
            emb_dropout=cfg.network.emb_dropout,
            pretrain=cfg.network.init,
            joint=cfg.network.joint,
            design_details=design_details
        )  # Must set jit=False for training  ViT-B/32
        return model
    return model
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.data.input_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
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
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.PROMPTSRC.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        # clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda() #TODO: Revise
        clip_model_temp = load_clip_to_cpu(cfg, True).cuda() #TODO: Revise
        clip_model_temp_image = load_clip_to_cpu(cfg, True) #TODO: Revise
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
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

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, image_features, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
       
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

            return F.cross_entropy(logits,
                                   label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits
    
    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))
    
#!##############################################################################
'''
class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
'''


def print_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


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

    design_details = {"trainer": 'IVLP',
                        "vision_depth": config.PROMPTSRC.PROMPT_DEPTH_VISION,
                        "language_depth": config.PROMPTSRC.PROMPT_DEPTH_TEXT,
                        "vision_ctx": config.PROMPTSRC.N_CTX_VISION,
                        "language_ctx": config.PROMPTSRC.N_CTX_TEXT}
    model, clip_state_dict = clip.load(
        config.network.arch,
        config,
        device=torch.device('cpu'),
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint=config.network.joint,
        design_details=design_details
    )  # Must set jit=False for training  ViT-B/32
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
    
    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    print("train transforms: {}".format(transform_train.transforms))
    print("val transforms: {}".format(transform_val.transforms))
    ############################## dataset  loader ###################################
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
    #################################################################
    #! Added#######
    classnames = [name for id, name in train_data.classes]
    customCLIP = CustomCLIP(config, classnames, model).to(device)
    print("Turning off gradients in both the image and the text encoder")
    for name, param in customCLIP.named_parameters():
        if "prompt_learner" not in name and "prompt" not in name and "Adapter" not in name: # EZ_CLIP + CoOp
            if "VPT" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        else:
            if "ZS_image_encoder" in name:
                param.requires_grad_(False)
    customCLIP = torch.nn.DataParallel(customCLIP, device_ids=[0]).cuda()
    #! ############
    ''' #! original
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    '''
    prompt_learner_param_count = 0
    param_count_with_prompts = 0
    param_count_with_adapters = 0
    visual_param_count_with_adapters = 0
    visual_param_count_with_T_adapters = 0
    VPT_param_count = 0
    for name, param in customCLIP.named_parameters():
        if "prompt_learner" in name:
            print("prompt_learner--", name)
            prompt_learner_param_count += param.numel()
        elif "prompt" in name:
            print("prompt--", name)
            param_count_with_prompts += param.numel()
        if "ZS_image_encoder" in name:
            prompt_learner_param_count -= param.numel()
            print("ZS_image_encoder--", name)
        if "Adapter" in name and not "visual" in name:
            print("text---", name)
            param_count_with_adapters += param.numel()
        if "Adapter" in name and "visual" in name:
            print("visual--", name)
            visual_param_count_with_adapters += param.numel()
        if "T_Adapter" in name and "visual" in name:
            print("temporal visual--", name)
            visual_param_count_with_T_adapters += param.numel()
        if "VPT" in name:
            print("VPT--", name)
            VPT_param_count += param.numel()
            
    param_count_with_prompts_in_million = param_count_with_prompts / 1_000_000
    param_count_with_adapter_in_million = (param_count_with_adapters) / 1_000_000
    T_visual_param_count_with_adapter_in_million = (
        visual_param_count_with_T_adapters / 1_000_000
    )
    S_visual_param_count_with_adapter_in_million = (
        visual_param_count_with_adapters - visual_param_count_with_T_adapters
    ) / 1_000_000
    prompt_learner_param_count_in_million = prompt_learner_param_count / 1_000_000
    VPT_param_count_in_million = VPT_param_count / 1_000_000
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
        f'Number of Trainable Parameters with "prompt_learner" in their names: {prompt_learner_param_count_in_million:.3f}'
    )
    print(
        f'Number of Trainable Parameters with "VPT" in their names: {VPT_param_count_in_million:.3f}'
    )
    
    '''
    for name, p in model.named_parameters():
        if "prompt" not in name and "Adapter" not in name:
            p.requires_grad = False
    '''
    ###########################################################
    parameters = filter(lambda p: p.requires_grad, customCLIP.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Modified CLIP_model Trainable Parameters: %.3fM" % parameters)

    '''#!###########################################################
    # prompt_vit_model = torch.nn.DataParallel(prompt_vit_model).cuda()
    model_text = torch.nn.DataParallel(model_text, device_ids=[0]).cuda()
    model_image = torch.nn.DataParallel(model_image, device_ids=[0]).cuda()
    

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text
        )  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)
    '''#!###########################################################
    '''
    loss_img = KLLoss()
    loss_txt = KLLoss()
    '''
    loss_img = nn.CrossEntropyLoss()
    loss_motion = Motion_loss()

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(
                (
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        config.evaluate, start_epoch
                    )
                )
            )
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))
    ''' #!###############################
    # classes: [num_text_aug, tensor([n_cls, n_tkn])]
    # text_dict:{key:index in text_aug, value: tensor([n_cls, n_tkn])}
    # train_data.classes: list[(id,name)]
    classes, num_text_aug, text_dict = text_prompt(
        train_data, config.data.gpt_discription, config.data.use_llm
    )
    ''' #!###############################
    
    optimizer = _optimizer(config, model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    loss = []
    top_1_acc = []
    ucf_top_1_acc = []
    ucfds_top_1_acc = []
    hmdb_top_1_acc = []
    k600_top_1_acc = []

    best_prec1 = 0.0
    ucf_best_prec1 = 0.0
    ucfds_best_prec1 = 0.0
    hmdb_best_prec1 = 0.0
    k600_best_prec1 = 0.0

    if config.solver.evaluate:
        prec1 = validate(
            start_epoch,
            val_loader,
            # classes,
            device,
            customCLIP,
            config,
            # num_text_aug,
            working_dir,
            config.data.dataset,
            is_Train=True,
        )
        return

    for k, v in model.named_parameters():
        if v.requires_grad:
            print("{}: {}".format(k, v.requires_grad))

    for epoch in range(start_epoch, config.solver.epochs):
        print(
            "------------------------------------------------------------------------"
        )
        print("Epoch %d start .." % epoch)
        '''
        model_image.train()
        model_text.train()
        '''
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
            '''
            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i, :] for i, j in zip(list_id, text_id)])
            '''
            prompt_images = prompt_images.view(
                -1, c, h, w
            )  # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            # texts = texts.to(device)
            image_embedding = customCLIP.module.encode_image(prompt_images)
            image_embedding = image_embedding.view(b, t, -1)
            if config.use_motion_loss:
                loss_video_motion = loss_motion(image_embedding)
            image_embedding = image_embedding.mean(dim=1, keepdim=False)


            '''
            text_embedding = customCLIP.module.encode_text()

            if config.network.fix_text:
                text_embedding.detach_()
            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(
                image_embedding, text_embedding, logit_scale
            )
            '''
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = customCLIP(prompt_images, image_embedding, list_id)
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.cuda(),
                                      reduction='mean') * config.PROMPTSRC.TEXT_LOSS_WEIGHT
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.cuda(),
                                       reduction='mean') * config.PROMPTSRC.IMAGE_LOSS_WEIGHT
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            loss = (loss_ce + L_SCL)
            '''
            ground_truth = torch.tensor(
                gen_label(list_id), dtype=image_embedding.dtype, device=device
            )
            '''
            #!To Remove##############################
            '''
            print(f'image_embedding.shape:{image_embedding.shape}, text_embedding.shape:{f.shape}')
            print(f'logits_per_image.shape:{logits_per_image.shape}, ground_truth.shape:{ground_truth.shape}')
            print(f'logits_per_text.shape:{logits_per_text.shape}, ground_truth.shape:{ground_truth.shape}')
            image_embedding.shape:torch.Size([16, 512]), text_embedding.shape:torch.Size([58, 512])
            logits_per_image.shape:torch.Size([16, 58]), ground_truth.shape:torch.Size([16, 16])
            logits_per_text.shape:torch.Size([58, 16]), ground_truth.shape:torch.Size([16, 16])
            '''
            #!##############################
            # loss_imgs = loss_img(logits_per_image, ground_truth)
            # loss_texts = loss_txt(logits_per_text, ground_truth)
            # list_id = torch.tensor(list_id).long().to(device=device)
            # loss_imgs = loss_img(logits_per_image, list_id)
            '''
            if config.use_motion_loss:
                total_loss = (loss_imgs + loss_texts) / 2 + loss_video_motion
            else:
                total_loss = (loss_imgs + loss_texts) / 2
            '''
            if config.use_motion_loss:
                total_loss = loss + loss_video_motion
            else:
                total_loss = loss
            
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
                        "Epoch:%d  iteration:%d/%d, total loss:%f, promptSRC loss:%f, motion loss:%f, lr:%f "
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss.item(),
                            # loss_texts.item(),
                            loss_video_motion.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    print(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, promptSRC loss:%f, lr:%f "
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss.item(),
                            # loss_texts.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            print("{} val accuracy".format(config.data.dataset))
            prec1 = validate(
                epoch,
                val_loader,
                # classes,
                device,
                customCLIP,
                config,
                # num_text_aug,
                working_dir,
                config.data.dataset,
                is_Train=True,
            )
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print("{} Testing: {}/{}".format(config.data.dataset, prec1, best_prec1))

        txt_path = "{}/log.txt".format(working_dir)
        if os.path.exists(txt_path):
            with open(txt_path, "a+") as f:
                f.write("\n")
                if config.use_motion_loss:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, promptSRC loss:%f, motion loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss.item(),
                            # loss_texts.item(),
                            loss_video_motion.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, promptSRC loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss.item(),
                            # loss_texts.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                f.write(
                    "{} Testing: {}/{}\n".format(config.data.dataset, prec1, best_prec1)
                )
                f.close()
        else:
            with open(txt_path, mode="wt") as f:
                if config.use_motion_loss:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, promptSRC loss:%f, motion loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss.item(),
                            # loss_texts.item(),
                            loss_video_motion.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    f.write(
                        "Epoch:%d  iteration:%d/%d, total loss:%f, promptSRC loss:%f, lr:%f \n"
                        % (
                            epoch,
                            kkk,
                            len(train_loader),
                            total_loss.item(),
                            loss.item(),
                            # loss_texts.item(),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                f.write(
                    "{} Testing: {}/{}\n".format(config.data.dataset, prec1, best_prec1)
                )
                f.close()

        print("Saving:")
        filename1 = "{}/last_model.pt".format(working_dir)
        # filename = "{}/epoch_{}_model.pt".format(working_dir,epoch)
        top_1_acc.append(prec1 / 100)
        loss.append(np.mean(epoch_loss))
        # epoch_saving(epoch, model,  optimizer, filename)
        epoch_saving(epoch, customCLIP, optimizer, filename1)
        if is_best:
            print(
                "Saving best weight based on {} accuracy at epoch {}".format(
                    config.data.dataset, epoch
                )
            )
            best_saving(working_dir, epoch, customCLIP, optimizer, config.data.dataset)

        print("Epoch %d end .." % epoch)
        ##############graph_plot################
        X = list(range(len(loss)))
        plt.plot(X, loss, color="r", label="Training loss")
        plt.plot(
            X, top_1_acc, color="g", label="{} Accuracy".format(config.data.dataset)
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
if __name__ == "__main__":
    main()
