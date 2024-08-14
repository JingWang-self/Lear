import os
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path as osp
from collections import OrderedDict
import math
# import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *

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

    def forward(self, prompts, tokenized_prompts): 
        #! prompts: the learnable context
        #! tokenized_prompts: tokenizerd text, including class infomation
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.COCOOP.N_CTX
        ctx_init = cfg.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim #! add
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.data.input_size
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(OrderedDict([ #! add, to be optimized
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if cfg.COCOOP.PREC == "fp16":
            self.meta_net.half()

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
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

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
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_features, label=None):

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        # logits: (batch, n_cls)
        return logits
    
    def encode_image(self, image):
        return  self.image_encoder(image.type(self.dtype))
    
#!##############################################################################


######### TSNE PLOT ##########
def feature_plot(video_feature, labels, working_dir, epoch, dataset_name):
    name = "embedding_%d.pdf" % (epoch)
    features = torch.cat(video_feature, dim=0)
    labels = torch.cat(labels, dim=0)
    # num_of_class = len(labels.unique())
    plot_dir = os.path.join(working_dir, dataset_name + "_plot")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
    tsne_result = tsne.fit_transform(features.cpu())
    plt.figure(figsize=(10, 8))
    plt.scatter(
        tsne_result[:, 0],
        tsne_result[:, 1],
        c=labels.cpu(),
        cmap=plt.get_cmap("tab20"),
        s=6,
    )
    plt.colorbar()
    plt.title("t-SNE Visualization")
    plt.savefig(os.path.join(plot_dir, name), format="pdf", dpi=300)
    plt.close()


###############################
def Class_analysis(
    cls_dic, pred, orignal, video_features, labels, working_dir, epoch, dataset_name
):
    plot_dir = os.path.join(working_dir, dataset_name + "_class_analysis")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    txt_path = "{}/class_analysis_{}.txt".format(plot_dir, epoch)
    plot_path = "{}/class_analysis_{}.pdf".format(plot_dir, epoch)
    confusion_matrix_plot_path = "{}/confusion_class_analysis_{}.pdf".format(
        plot_dir, epoch
    )
    cls_list = []
    percentage_list = []
    cls_persentage_dic = {}
    with open(txt_path, mode="wt") as f:
        for key in np.sort(list(cls_dic.keys())):
            f.write("\n")
            alist = cls_dic[key]
            percentage_of_ones = (sum(alist) / len(alist)) * 100
            cls_list.append(key)
            percentage_list.append(percentage_of_ones)
            cls_persentage_dic.update({key: percentage_of_ones})
            f.write("class: {}---{}\n".format(key, percentage_of_ones))
        f.close()
    sorted_dict = {
        k: v
        for k, v in sorted(
            cls_persentage_dic.items(), key=lambda item: item[1], reverse=True
        )
    }
    top_30_keys = list(sorted_dict.keys())[:30]
    label_new = []
    video_features_new = []
    for label, feature in zip(labels, video_features):
        if label.item() in top_30_keys:
            label_new.append(label)
            video_features_new.append(feature)
    # print(len(label_new), len(video_features_new))
    feature_plot(video_features_new, label_new, working_dir, epoch, dataset_name)
    plt.bar(cls_list, percentage_list)
    # Set labels and title
    plt.xlabel("Class Numbers")
    plt.ylabel("Percentage (%)")
    # plt.title('Class Analysis {}'.format(epoch))
    plt.title("Class Analysis")
    plt.savefig(plot_path, format="pdf", dpi=300)
    plt.close()

    cm = confusion_matrix(orignal, pred)
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", cbar=True)

    # Set labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix on {}".format(dataset_name))
    plt.savefig(confusion_matrix_plot_path, format="pdf", dpi=300)
    plt.close()


###############################
def plot_image_classification(ax, image, predictions, true_label, labels2name, top_k=5):
    """
    Function to plot the image with the top-k predictions.

    Parameters:
    - ax: The subplot axis to draw the image on
    - image: The image tensor
    - predictions: List of tuples (class_name, probability)
    - true_label: True label of the image
    - labels2name: Dictionary to convert label ids to names
    - top_k: Number of top predictions to display
    """
    img = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)
    ax.axis("off")

    ax.set_title(f"True label: {labels2name[true_label]}", color="green")

    y_offset = img.shape[0] * 0.1
    for i, (class_id, prob) in enumerate(predictions[:top_k]):
        ax.text(
            0,
            img.shape[0] + y_offset * (i + 1),
            f"{labels2name[class_id]}: {prob:.2f}%",
            color="blue",
        )
#! Added#######################
def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res
#!#############################


def validate(
    epoch,
    val_loader,
    # classes, #! to remove
    device,
    model,
    config,
    # num_text_aug, #! to remove
    working_dir,
    dataset_name,
    labels2name=None,
    is_Train=False,
):
    model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    cls_dic = {}
    video_features = []
    labels = []
    pred = []
    orignal = []
    image_batch = []
    print(f"Test saving in: {working_dir}")
    with torch.no_grad():
        # text_inputs = classes.to(device)
        # text_features = model.encode_text(text_inputs)
        for iii, (prompt_image, class_id) in enumerate(tqdm(val_loader)):
            prompt_image = prompt_image.view(
                (-1, config.data.num_segments, 3) + prompt_image.size()[-2:]
            )
            b, t, c, h, w = prompt_image.size()
            class_id = class_id.to(device)
            image_input = prompt_image.to(device).view(-1, c, h, w)
            image_features = model.module.encode_image(image_input).view(b, t, -1)
            image_features = image_features.mean(dim=1, keepdim=False)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = model(image_features, class_id)
            similarity = similarity.view(b, -1).softmax(dim=-1)
            class_for_plot = random.sample(range(230), 160)
            for id, feature in zip(class_id, image_features):
                if dataset_name == "K600":
                    if id.item() in class_for_plot:
                        video_features.append(feature.unsqueeze(0))
                        labels.append(torch.tensor([id.item()]))
                else:
                    video_features.append(feature.unsqueeze(0))
                    labels.append(torch.tensor([id.item()]))
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            pred.append(indices_1.squeeze().tolist())
            orignal.append(class_id.squeeze().tolist())
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item(): [1]})
                    else:
                        cls_dic[class_id[i].item()].append(1)
                else:
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item(): [0]})
                    else:
                        cls_dic[class_id[i].item()].append(0)

                if class_id[i] in indices_5[i]:
                    corr_5 += 1
            
                # Save image for visualization
                if len(image_batch) < 6:
                    image_batch.append((image_input[i], class_id[i], similarity[i]))
            
            '''
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            pred.append(indices_1.squeeze().tolist())
            orignal.append(class_id.squeeze().tolist())
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item(): [1]})
                    else:
                        cls_dic[class_id[i].item()].append(1)
                else:
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item(): [0]})
                    else:
                        cls_dic[class_id[i].item()].append(0)

                if class_id[i] in indices_5[i]:
                    corr_5 += 1
            
                # Save image for visualization
                if len(image_batch) < 6:
                    image_batch.append((image_input[i], class_id[i], similarity[i]))
            '''
            

    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100

    if not is_Train:
        pred = [item for sublist in pred for item in sublist]
        orignal = [item for sublist in orignal for item in sublist]
        Class_analysis(
            cls_dic,
            pred,
            orignal,
            video_features,
            labels,
            working_dir,
            epoch,
            dataset_name,
        )
        feature_plot(video_features, labels, working_dir, epoch, dataset_name)

        '''
        # Plot the images and their predictions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (image, true_label, similarities) in enumerate(image_batch):
            ax = axes[i]
            predictions = [
                (idx, prob) for idx, prob in enumerate(similarities.cpu().numpy())
            ]
            predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

            plot_image_classification(
                ax, image, predictions, true_label.item(), labels2name
            )

        plt.tight_layout()
        plt.savefig(
            f"{working_dir}/{dataset_name}_batch_visualization_{epoch}.pdf",
            format="pdf",
            dpi=300,
        )
        plt.close()
        '''

    print(
        "Epoch: [{}/{}]: Top1: {}, Top5: {}".format(
            epoch, config.solver.epochs, top1, top5
        )
    )
    return top1


def validate_1(
    epoch,
    val_loader,
    classes,
    device,
    model,
    config,
    num_text_aug,
    working_dir,
    dataset_name,
    is_Train=False,
):
    model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    cls_dic = {}
    video_features = []
    labels = []
    pred = []
    orignal = []
    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.module.encode_text(text_inputs)
        for iii, (prompt_image, class_id) in enumerate(tqdm(val_loader)):
            prompt_image = prompt_image.view(
                (-1, config.data.num_segments, 3) + prompt_image.size()[-2:]
            )
            b, t, c, h, w = prompt_image.size()
            class_id = class_id.to(device)
            image_input = prompt_image.to(device).view(-1, c, h, w)
            image_features = model.module.encode_image(image_input).view(b, t, -1)
            image_features = image_features.mean(dim=1, keepdim=False)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            class_for_plot = random.sample(range(230), 160)
            for id, feature in zip(class_id, image_features):
                if dataset_name == "K600":
                    if id.item() in class_for_plot:
                        video_features.append(feature.unsqueeze(0))
                        labels.append(torch.tensor([id.item()]))
                else:
                    video_features.append(feature.unsqueeze(0))
                    labels.append(torch.tensor([id.item()]))
            similarity = 100.0 * image_features @ text_features.T
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            # print(values_1, indices_1)
            pred.append(indices_1.squeeze().tolist())
            orignal.append(class_id.squeeze().tolist())
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item(): [1]})
                    else:
                        cls_dic[class_id[i].item()].append(1)
                else:
                    if class_id[i].item() not in list(cls_dic.keys()):
                        cls_dic.update({class_id[i].item(): [0]})
                    else:
                        cls_dic[class_id[i].item()].append(0)

                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    pred = [item for sublist in pred for item in sublist]
    orignal = [item for sublist in orignal for item in sublist]
    print(
        "Epoch: [{}/{}]: Top1: {}, Top5: {}".format(
            epoch, config.solver.epochs, top1, top5
        )
    )
    # Class_analysis(cls_dic,pred,orignal, video_features,labels, working_dir,epoch,dataset_name)
    # feature_plot(video_features,labels,working_dir,epoch,dataset_name)

    return top1


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-cfg", default="configs/ucf101/UCF_zero_shot_testing.yaml"
    )
    parser.add_argument("--traning_name", default="")
    args = parser.parse_args()
    args.traning_name = "SSv2_base_to_novel_testing"  # give testing name
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
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

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(
        config.network.arch,
        config,
        device='cpu',
        jit=False,
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
    )  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)
    '''
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    for name, p in model.named_parameters():
        if "prompt" not in name and "Adapter" not in name:
            p.requires_grad = False
    '''


    val_data = Action_DATASETS(
        config.data.val_list,
        config.data.label_list,
        num_segments=config.data.num_segments,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val,
        random_shift=config.random_shift,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    classnames = [name for id, name in val_data.classes]
    customCLIP = CustomCLIP(config, classnames, model).to(device)
    print("Turning off gradients in both the image and the text encoder")
    for name, param in customCLIP.named_parameters():
        if "prompt_learner" not in name and "prompt" not in name and "Adapter" not in name:
            param.requires_grad_(False)
    customCLIP = torch.nn.DataParallel(customCLIP, device_ids=[0]).cuda()
    '''
    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text
        )  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)
    '''

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            customCLIP.load_state_dict(checkpoint["model_state_dict"], strict=False)
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))
    '''
    classes, num_text_aug, text_dict = text_prompt(
        val_data, config.data.gpt_discription, config.data.use_llm
    )
    '''

    best_prec1 = 0.0
    # 'UCF_base' is dataset name

    # labels2name: dict {'0': 'xxx',}
    labels_csv_path = config.data.label_list
    with open(labels_csv_path, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        labels2name = {int(row[0]): row[1] for row in reader}
    print(labels2name)

    prec1 = validate(
        start_epoch,
        val_loader,
        # classes,
        device,
        customCLIP,
        config,
        # num_text_aug,
        working_dir,
        config["data"]["dataset"],
        labels2name,
        is_Train=False,
    )


if __name__ == "__main__":
    main()
