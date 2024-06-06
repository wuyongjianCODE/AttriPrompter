# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized VL R-CNN framework
"""
import os.path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.utils import cat, permute_and_flatten
from ..backbone import build_backbone
from ..rpn import build_rpn
from ..roi_heads import build_roi_heads

from ..language_backbone import build_language_backbone
from transformers import AutoTokenizer

import random
import timeit
import pdb
from copy import deepcopy
import numpy
def cos_similarity(tensorx):
    sim_=torch.matmul(tensorx,tensorx.transpose(-1,-2)).cpu().numpy()
    sim_range01=sim_.copy()
    for i in range(sim_.shape[-2]):
        for j in range(sim_.shape[-1]):
            sim_range01[i,j]/=numpy.sqrt(sim_[i,i])
            sim_range01[i, j] /= numpy.sqrt(sim_[j, j])
    return sim_range01
def cos_similarity2(tensorx,tensorb):
    sim_=torch.matmul(tensorx,tensorb.transpose(-1,-2)).cpu().numpy()
    sim_range01=sim_.copy()
    # for i in range(sim_.shape[-2]):
    #     for j in range(sim_.shape[-1]):
    #         sim_range01[i,j]/=numpy.sqrt(sim_[i,i])
    #         sim_range01[i, j] /= numpy.sqrt(sim_[j, j])
    return sim_range01
def random_word(input_ids, mask_token_id, vocabs, padding_token_id, greenlight_map):
    """
    greenlight_map, batch_size x 256 (seq_len):
        0 means this location cannot be calculated in the MLM loss
        -1 means this location cannot be masked!!
        1 means this location can be masked and can be calculated in the MLM loss
    """
    output_label = deepcopy(input_ids)
    for j in range(input_ids.size(0)):
        for i in range(input_ids.size(1)):
            prob = random.random()
            # mask token with probability
            ratio = 0.15
            if greenlight_map is not None and greenlight_map[j, i] == -1:
                output_label[j, i] = -100
                continue

            if (not input_ids[j, i] == padding_token_id) and prob < ratio:
                prob /= ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    input_ids[j, i] = mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    input_ids[j, i] = random.choice(vocabs)

            else:
                # no masking token (will be ignored by loss function later)
                output_label[j, i] = -100

            if greenlight_map is not None and greenlight_map[j, i] != 1:
                output_label[j, i] = -100  # If this location should not be masked
    return input_ids, output_label


def regenerate_positive_map(tokenizer_input, positive_map):
    positive_map.clear()
    input_ids = tokenizer_input['input_ids'][tokenizer_input['input_ids'] != 0]
    phrase_count = 0
    positions_list_of_current_phrase = []
    for id, t_num in enumerate(input_ids):
        if phrase_count >= 79:
            break
        if t_num == 101:
            continue
        elif t_num == 1012 or t_num == 102:
            if len(positions_list_of_current_phrase) > 0:
                phrase_count += 1
                positive_map.update({phrase_count: positions_list_of_current_phrase})
                positions_list_of_current_phrase = []
        else:
            positions_list_of_current_phrase.append(id)
    return positive_map


class conv_map_module(nn.Module):
    """ map support image feature (5-stage) to text space."""

    def __init__(self, in_channels, out_channels, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(100,100))
        self.conv2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(50,50))
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(25,25))
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(13, 13))
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(7, 7))
        self.drop = nn.Dropout(drop)
        self.fc=nn.Linear(5*in_channels,768)
        self.act = act_layer(inplace=True)
    def forward(self, x):
        x1 = self.conv1(x[0])
        x2 = self.conv2(x[1])
        x3 = self.conv3(x[2])
        x4 = self.conv4(x[3])
        x5 = self.conv5(x[4])
        x=cat((x1,x2,x3,x4,x5),dim=0)
        x=self.fc(x.flatten())
        x = self.act(x)
        x = self.drop(x)
        return x


class GeneralizedVLRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedVLRCNN, self).__init__()
        # if cfg.IMPROMPT.gvl:
        #     cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM=13343
        self.cfg = cfg
        # visual encoder
        self.backbone = build_backbone(cfg)

        # language encoder
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            # self.tokenizer = build_tokenizer("clip")
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                print("Reuse token 'ðŁĴĳ</w>' (token_id = 49404) for mask token!")
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                   from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                                   from_slow=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [item for key, item in self.tokenizer_vocab.items()]

        self.language_backbone = build_language_backbone(cfg)

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.DEBUG = cfg.MODEL.DEBUG
        if abs(cfg.IMPROMPT.gvl)==1:
            self.reference_image_tensor=None
            self.reference_gt=None
            self.reference_length=0
            self.reference_length_make_sure_same_dataset=0
            self.reference_image_tensor_make_sure_same_dataset = None
        self.freeze_backbone = cfg.MODEL.BACKBONE.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER

        class Mlp(nn.Module):
            """ Multilayer perceptron."""

            def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
                super().__init__()
                out_features = out_features or in_features
                hidden_features = hidden_features or in_features
                self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
                self.act = act_layer(inplace=True)
                self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
                self.drop = nn.Dropout(drop)

            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.act(x)  # caution: remember uncomment to rollback
                x = self.drop(x)
                return x
        if self.cfg.generalized_vl:
            self.adapter = []
            for i in range(5):
                self.adapter.append(nn.Conv2d(256, 256, kernel_size=1, stride=1))

            mlp_v2=True# caution: remember change to false to rollback!!!!!!!!!!!!!!!!!!!!
            cin = 256 * 7 * 7
            cout=int(0.5*cin)
            if mlp_v2:
                self.my_fc= nn.Sequential(
                    Mlp(cin,cout,cin),
                )
            else:
                self.my_fc = nn.Sequential(
                    nn.Linear(cin, cout, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(cout, cin, bias=False),
                    nn.ReLU(inplace=True)
                )
        ##############self.my_fc=nn.Linear(4096,4096,bias=False)
        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES
        if cfg.IMPROMPT.gvl==-1:
            self.adaptivepool_of_stages=[]
            self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.adaptivepool_of_stages.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.adaptivepool_of_stages.append(nn.Identity())
            if cfg.IMPROMPT.map_module=='mlp':
                self.fcto768=Mlp(in_features=49*256,hidden_features=768,out_features=768)
            elif cfg.IMPROMPT.map_module=='conv':
                self.fcto768=conv_map_module(256,256)
        from einops import rearrange
        class CrossAttention(nn.Module):
            def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
                super(CrossAttention, self).__init__()
                self.emb_dim = emb_dim
                self.scale = emb_dim ** -0.5

                self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

                self.Wq = nn.Linear(emb_dim, emb_dim)
                self.Wk = nn.Linear(emb_dim, emb_dim)
                self.Wv = nn.Linear(emb_dim, emb_dim)

                self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

            def forward(self, x, context, pad_mask=None):
                '''

                :param x: [batch_size, c, h, w]
                :param context: [batch_szie, seq_len, emb_dim]
                :param pad_mask: [batch_size, seq_len, seq_len]
                :return:
                '''
                b, c, h, w = x.shape
                with torch.autograd.set_detect_anomaly(True):
                    x2 = self.proj_in(x)  # [batch_size, c, h, w] = [3, 512, 512, 512]
                    #x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, c] = [3, 262144, 512]
                    x3= x2.permute(0, 2,3,1)
                    x4=x3.view(b,h*w,self.emb_dim)

                    Q = self.Wq(x4)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
                    K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
                    V = self.Wv(context)

                    # [batch_size, h*w, seq_len]
                    att_weights = torch.einsum('bid,bjd -> bij', Q, K)
                    att_weights = att_weights * self.scale

                    if pad_mask is not None:
                        # [batch_size, h*w, seq_len]
                        att_weights = att_weights.masked_fill(pad_mask, -1e9)

                    att_weights = F.softmax(att_weights, dim=-1)
                    out = torch.einsum('bij, bjd -> bid', att_weights, V)  # [batch_size, h*w, emb_dim]
                    out = out.permute(0,2,1)
                    out=out.view(b,self.emb_dim,h,w)
                    # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, c, h, w]
                    out = self.proj_out(out)  # [batch_size, c, h, w]

                # print(out.shape)

                return x, att_weights
        self.vl_cross_att=cfg.vl_cross_att
        if cfg.vl_cross_att or cfg.apoadapter:
            print('vl_cross_att equipped!!!!!!!!!!!')
            self.cross_attention1= CrossAttention(256,768)
            self.cross_attention2 = CrossAttention(256, 768)
            self.cross_attention3 = CrossAttention(256, 768)
            self.cross_attention4 = CrossAttention(256, 768)
            self.cross_attention5 = CrossAttention(256, 768)
        if cfg.apoadapter:
            self.cross_attention_l2v=CrossAttention(768,256)
        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if hasattr(self.backbone, 'fpn'):
                assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, 'cls_logits'):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_language_backbone = self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE
        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.parameters():
                p.requires_grad = False

        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

        if self.cfg.GLIPKNOW.KNOWLEDGE_FILE:
            from maskrcnn_benchmark.data.datasets.tsv import load_from_yaml_file
            self.class_name_to_knowledge = load_from_yaml_file(self.cfg.GLIPKNOW.KNOWLEDGE_FILE)
            self.class_name_list = sorted([k for k in self.class_name_to_knowledge])

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GeneralizedVLRCNN, self).train(mode)
        if self.freeze_backbone:
            self.backbone.body.eval()
            for p in self.backbone.body.parameters():
                p.requires_grad = False
        if self.freeze_fpn:
            self.backbone.fpn.eval()
            for p in self.backbone.fpn.parameters():
                p.requires_grad = False
        if self.freeze_rpn:
            if hasattr(self.rpn, 'head'):
                self.rpn.head.eval()
            for p in self.rpn.parameters():
                p.requires_grad = False
        if self.linear_prob:
            if self.rpn is not None:
                for key, value in self.rpn.named_parameters():
                    if not (
                            'bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
            if self.roi_heads is not None:
                for key, value in self.roi_heads.named_parameters():
                    if not (
                            'bbox_pred' in key or 'cls_logits' in key or 'centerness' in key or 'cosine_scale' in key or 'dot_product_projection_text' in key or 'head.log_scale' in key or 'head.bias_lang' in key or 'head.bias0' in key):
                        value.requires_grad = False
        if self.freeze_cls_logits:
            if hasattr(self.rpn.head, 'cls_logits'):
                self.rpn.head.cls_logits.eval()
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False
        if self.add_linear_layer:
            if self.rpn is not None:
                for key, p in self.rpn.named_parameters():
                    if 'tunable_linear' in key:
                        p.requires_grad = True

        if self.freeze_language_backbone:
            self.language_backbone.eval()
            for p in self.language_backbone.parameters():
                p.requires_grad = False

    def forward(self,
                images,
                targets=None,
                captions=None,
                positive_map=None,
                greenlight_map=None,
                reference_map=None
                ):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

            mask_black_list: batch x 256, indicates whether or not a certain token is maskable or not

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # if self.cfg.print_flops:
        #     self.training=True
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        # batch_size = images.tensors.shape[0]
        device = images.tensors.device

        if self.cfg.GLIPKNOW.PARALLEL_LANGUAGE_INPUT:
            language_dict_features, positive_map = self._forward_language_parallel(
                captions=captions, targets=targets, device=device,
                positive_map=positive_map)
        else:
            # language embedding
            language_dict_features = {}
            if captions is not None:
                # print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions,
                                                             max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                             padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                             return_special_tokens_mask=True,
                                                             return_tensors='pt',
                                                             truncation=True).to(device)
                if self.use_mlm_loss:
                    if not self.mlm_loss_for_only_positives:
                        greenlight_map = None
                    input_ids, mlm_labels = random_word(
                        input_ids=tokenized.input_ids,
                        mask_token_id=self.tokenizer.mask_token_id,
                        vocabs=self.tokenizer_vocab_ids,
                        padding_token_id=self.tokenizer.pad_token_id,
                        greenlight_map=greenlight_map)
                else:
                    input_ids = tokenized.input_ids
                    mlm_labels = None

                tokenizer_input = {"input_ids": input_ids,
                                   "attention_mask": tokenized.attention_mask}
                if self.cfg.use_bitfit:
                    for name, param in self.language_backbone.named_parameters():
                        if 'bias' not in name :
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                    for name, param in self.backbone.named_parameters():
                        if 'bias' not in name :
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                if self.cfg.use_bitfit_cross:
                    for name, param in self.named_parameters():
                        if 'bias' in name and 'backbone' not in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
                    if self.cfg.FROZEE_BERT:
                        for name, param in self.language_backbone.named_parameters():
                            if 'adapter' not in name and 'lora_' not in name:
                                param.requires_grad = False
                            else:
                                param.requires_grad = True
                        if self.cfg.use_lora_text:
                            import loralib as lora
                            lora.mark_only_lora_as_trainable(self)
                        language_dict_features = self.language_backbone(tokenizer_input)
                    else:
                        language_dict_features = self.language_backbone(tokenizer_input)
                else:
                    if self.cfg.FROZEE_BERT:
                        for name, param in self.language_backbone.named_parameters():
                            if 'adapter' not in name: ##wuyongjian edit: here is for the text adapter ABC
                                param.requires_grad = False
                            else:
                                param.requires_grad = True
                        if self.cfg.use_lora_text:
                            import loralib as lora
                            lora.mark_only_lora_as_trainable(self)
                        language_dict_features = self.language_backbone(tokenizer_input)
                    else:
                        language_dict_features = self.language_backbone(tokenizer_input)
                if self.cfg.use_maple:
                    language_dict_features,maple_project_out=language_dict_features
                if not self.training:
                    positive_map=regenerate_positive_map(tokenizer_input,positive_map)
                # ONE HOT
                if self.cfg.DATASETS.ONE_HOT:
                    new_masks = torch.zeros_like(language_dict_features['masks'],
                                                 device=language_dict_features['masks'].device)
                    new_masks[:, :self.cfg.MODEL.DYHEAD.NUM_CLASSES] = 1
                    language_dict_features['masks'] = new_masks

                # MASK ALL SPECIAL TOKENS
                if self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL:
                    language_dict_features["masks"] = 1 - tokenized.special_tokens_mask

                language_dict_features["mlm_labels"] = mlm_labels

        # visual embedding
        G_vl=self.cfg.generalized_vl
        if G_vl:
            swint_feature_c4 = None
            FROZEE_SWINT = True
            USE_ADAPTER = True
            USE_ADAPTER_conv11 = False
        else:
            swint_feature_c4 = None
            FROZEE_SWINT = self.cfg.FROZEE_SWINT
            USE_ADAPTER = False
            USE_ADAPTER_conv11 = False
        if FROZEE_SWINT:
            # with torch.no_grad():  # wyj : add to freeze visual backbone!!!!!!!!!!!!!!!!!!!
            for name,param in self.backbone.named_parameters():
                if 'adapter' not in name and 'lora_' not in name:
                    param.requires_grad=False
                else:
                    param.requires_grad = True
                    # if 'layers.1' in name :#or 'layers.3' in name:
                    #     # print(name+'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    #     param.requires_grad = True
                    # else:
                    #     param.requires_grad = False
            # if self.cfg.use_lora_visual:
            #     import loralib as lora
            #     lora.mark_only_lora_as_trainable(self)
            if 'vl' in self.cfg.MODEL.SWINT.VERSION:
                # the backbone only updates the "hidden" field in language_dict_features
                inputs = {"img": images.tensors, "lang": language_dict_features}
                visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
            else:
                visual_features = self.backbone(images.tensors)
        else:
            if 'vl' in self.cfg.MODEL.SWINT.VERSION:
                # the backbone only updates the "hidden" field in language_dict_features
                inputs = {"img": images.tensors, "lang": language_dict_features}
                visual_features, language_dict_features, swint_feature_c4 = self.backbone(inputs)
            else:
                if self.cfg.use_maple:
                    visual_features = self.backbone((images.tensors,maple_project_out.view(-1,192,1,1)))
                else:
                    visual_features = self.backbone(images.tensors)
        if USE_ADAPTER:
            v = []
            self.my_fc.to(device)
            for id, tensori in enumerate(visual_features):
                # tensori=tensori.to(device)
                # visual_features[id]=self.adapter[id](tensori)
                # self.adapter[id].to(device)
                # v.append(self.adapter[id](tensori))
                if id != 4:
                    v.append(tensori)
                else:
                    tensori = tensori.flatten()
                    tensori = self.my_fc(tensori) + tensori
                    tensori = tensori.reshape([1, 256, 7, 7])
                    v.append(tensori)
            visual_features = v
        if USE_ADAPTER_conv11:
            v = []
            for id, tensori in enumerate(visual_features):
                self.adapter[id].to(device)
                v.append(tensori + self.adapter[id](tensori))
            visual_features = v
        # rpn force boxes
        if targets:
            targets = [target.to(device)
                       for target in targets if target is not None]
        if self.vl_cross_att>1:
            borrow_embedding=language_dict_features['embedded'].clone()
            cross_out0, att_out0= self.cross_attention1(visual_features[0].clone(),borrow_embedding)
            cross_out1, att_out1= self.cross_attention2(visual_features[1].clone(), borrow_embedding)
            cross_out2, att_out2= self.cross_attention3(visual_features[2].clone(), borrow_embedding)
            cross_out3, att_out3= self.cross_attention4(visual_features[3].clone(), borrow_embedding)
            cross_out4, att_out4= self.cross_attention5(visual_features[4].clone(), borrow_embedding)
            visual_features=[cross_out0,cross_out1,cross_out2,cross_out3,cross_out4]
        if self.cfg.apoadapter:
            borrow_embedding=language_dict_features['embedded'].clone()
            cross_out0, att_out0= self.cross_attention1(visual_features[0].clone(),borrow_embedding)
            cross_out1, att_out1= self.cross_attention2(visual_features[1].clone(), borrow_embedding)
            cross_out2, att_out2= self.cross_attention3(visual_features[2].clone(), borrow_embedding)
            cross_out3, att_out3= self.cross_attention4(visual_features[3].clone(), borrow_embedding)
            cross_out4, att_out4= self.cross_attention5(visual_features[4].clone(), borrow_embedding)
            language_dict_features['embedded']=self.cross_attention_l2v(borrow_embedding,visual_features[4].clone())
            visual_features=[cross_out0,cross_out1,cross_out2,cross_out3,cross_out4]
        if self.force_boxes:
            proposals = []
            for t in targets:
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                proposals.append(tb)
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                _, proposal_losses, fused_visual_features = self.rpn(
                    images, visual_features, targets, language_dict_features,
                    positive_map, captions, swint_feature_c4)
            elif self.training:
                null_loss = 0
                for key, param in self.rpn.named_parameters():
                    null_loss += 0.0 * param.sum()
                proposal_losses = {('rpn_null_loss', null_loss)}
        else:
            if abs(self.cfg.IMPROMPT.gvl)==1:
                # import cv2
                # from torchvision import transforms
                # image_path = "/home/data/jy/GLIP/croppedfil/341/8.jpg"  # 替换为你自己的图像路径
                # image = cv2.imread(image_path)
                #
                # # Define transformations
                # transform = transforms.Compose([
                #     transforms.ToPILImage(),
                #     transforms.Resize(800),
                #     transforms.ToTensor()
                # ])
                #
                # # 转换图像为张量并调整大小
                # resized_tensor = transform(image)
                if self.cfg.IMPROMPT.input_way=='first_gt_box':
                    if reference_map is not None:
                        reference_map_tensor=reference_map.unsqueeze(dim=0).cuda()
                    elif self.reference_image_tensor is not None:
                        reference_map_tensor=self.reference_image_tensor
                        reference_gt=self.reference_gt
                    else:
                        from torchvision import transforms
                        import copy
                        imprompt = copy.deepcopy(images)
                        impromptcrop = imprompt.tensors[0][:, 10:130, 75:180]
                        # Define transformations
                        transform = transforms.Compose([
                            transforms.Resize((800,800)),
                        ])
                        # 转换图像为张量并调整大小
                        resized_tensor = transform(impromptcrop)
                        # imprompt.tensors = resized_tensor
                        self.reference_image_tensor=resized_tensor.unsqueeze(dim=0).cuda()
                        reference_map_tensor = self.reference_image_tensor
                    reference_embeding = self.backbone(reference_map_tensor)
                elif self.cfg.IMPROMPT.input_way=='input_image_itself':
                    if self.training:
                        if self.reference_image_tensor is None or (self.cfg.INPUT.FIX_RES and self.training):
                            self.reference_image_tensor=[]
                            import copy
                            imprompt = copy.deepcopy(images)
                            from ..evaluation_utils import img_preprocess
                            bbs = targets[0].bbox
                            self.reference_length=bbs.shape[0]
                            for box_id in range(bbs.shape[0]):
                                mask_of_reference=np.zeros((images.tensors[0].shape[-2],images.tensors[0].shape[-1]))
                                THIS_BOX=bbs[box_id,:]
                                x1,y1,x2,y2=THIS_BOX.cpu()
                                mask_of_reference[int(np.round(y1)):int(np.round(y2)),int(np.round(x1)):int(np.round(x2))]=1
                                out_reference_image=[img_preprocess((None, [imprompt.tensors[0,:,:,:]], [mask_of_reference]), blur=3, bg_fac=0.1).numpy()[0]]
                                out_tensor=torch.from_numpy(out_reference_image[0])#.permute(1,2,0).numpy()
                                self.reference_image_tensor.append(out_tensor.unsqueeze(0).cuda().type(torch.float32))#caution:it is required to make sure data type
                        selected_reference_ID=random.randint(0,self.reference_length-1)
                        output_to_feed_to_backbone=self.reference_image_tensor[selected_reference_ID]
                        keep = []
                        if self.cfg.IMPROMPT.embed_768_to_lang_adaptively:
                            # TODO: change positive map
                            if torch.is_tensor(positive_map):
                                if self.cfg.IMPROMPT.train_gt_1_per_class == 0:
                                    selected_positive_map = torch.arange(positive_map.size(0)) < 0
                                    selected_positive_map[selected_reference_ID] = True
                                    positive_map = positive_map[selected_positive_map]
                                elif self.cfg.IMPROMPT.train_gt_1_per_class > 0:
                                    selected_positive_map = torch.arange(
                                        positive_map.size(0)) < 0  # wuyongjian: init an all_False mask
                                    for boxi, box_label in enumerate(targets[0].extra_fields['labels']):
                                        if box_label == targets[0].extra_fields['labels'][selected_reference_ID]:
                                            selected_positive_map[boxi] = True
                                    positive_map = positive_map[selected_positive_map]
                        if self.cfg.IMPROMPT.train_gt_1_per_class==0:
                            keep.append(selected_reference_ID)
                            targets[0] = targets[0].__getitem__(keep, restricted=True)
                        elif self.cfg.IMPROMPT.train_gt_1_per_class>0:
                            for boxi,box_label in enumerate(targets[0].extra_fields['labels']):
                                if box_label==targets[0].extra_fields['labels'][selected_reference_ID]:
                                    keep.append(boxi)
                            targets[0] = targets[0].__getitem__(keep, restricted=True)

                    else:#wuyongjian: when test, it maybe ccrcc to eval_1shot ,or coco to check if training time long enough
                        if self.reference_image_tensor_make_sure_same_dataset is None:
                            self.reference_image_tensor_make_sure_same_dataset = []
                            import copy
                            imprompt = copy.deepcopy(images)
                            from ..evaluation_utils import img_preprocess
                            bbs = reference_map[0].bbox
                            self.reference_length_make_sure_same_dataset = bbs.shape[0]
                            for box_id in range(bbs.shape[0]):
                                mask_of_reference=np.zeros((images.tensors[0].shape[-2],images.tensors[0].shape[-1]))
                                THIS_BOX = bbs[box_id, :]
                                x1, y1, x2, y2 = THIS_BOX.cpu()
                                mask_of_reference[int(np.round(y1)):int(np.round(y2)),
                                int(np.round(x1)):int(np.round(x2))] = 1
                                out_reference_image = [
                                    img_preprocess((None, [imprompt.tensors[0, :, :, :]], [mask_of_reference]), blur=3,
                                                   bg_fac=0.1).numpy()[0]]
                                out_tensor = torch.from_numpy(out_reference_image[0])  # .permute(1,2,0).numpy()
                                self.reference_image_tensor_make_sure_same_dataset.append(out_tensor.unsqueeze(0).cuda().type(
                                    torch.float32))  # caution:it is required to make sure data type
                        selected_reference_ID=random.randint(0, self.reference_length_make_sure_same_dataset - 1)
                        output_to_feed_to_backbone = self.reference_image_tensor_make_sure_same_dataset[selected_reference_ID]
                    reference_embeding = self.backbone(output_to_feed_to_backbone)
                elif self.cfg.IMPROMPT.input_way=='real_gt' :#and self.training:
                    from torchvision import transforms
                    import copy
                    imprompt = copy.deepcopy(images)
                    if targets is not None:
                        box_label=targets[0].bbox[0,:]
                        def tensor2int(x):
                            floatx=x.item()
                            if floatx-int(floatx)>0.5:
                                return int(floatx)+1
                            else:
                                return int(floatx)
                        x1=tensor2int(box_label[0])
                        y1 = max(tensor2int(box_label[1]),1)
                        x2 = tensor2int(box_label[2])
                        y2 = max(tensor2int(box_label[3]),1)
                        impromptcrop = imprompt.tensors[0,:,y1:y2, x1:x2]
                        # Define transformations
                        transform = transforms.Compose([
                            transforms.Resize((800, 800)),
                        ])
                        # 转换图像为张量并调整大小
                        resized_tensor = transform(impromptcrop)
                        # imprompt.tensors = resized_tensor
                        self.reference_image_tensor = resized_tensor.unsqueeze(dim=0).cuda()
                        reference_map_tensor = self.reference_image_tensor
                    else:#in test phrase, there is no _targets_ , so use the last gtbox (of a train im) to apply to this test im
                        reference_map_tensor = self.reference_image_tensor
                    reference_embeding = self.backbone(reference_map_tensor)
                visual_embeding_flatten=[]
                for ii, feat_per_level in enumerate(reference_embeding[-1:]):
                    # size_per_level.append([h, w])
                    feat_per_level=self.adaptivepool_of_stages[ii](feat_per_level)#wuyongjian: make sure it become 7x7
                    bs, c, h, w = feat_per_level.shape
                    feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
                    visual_embeding_flatten.append(feat)
                visual_embeding_flatten = cat(visual_embeding_flatten, dim=1)
                # visual_embeding_flatten = visual_embeding_flatten.permute(0, 2, 1)

                #######################################################embed_768_to_lang start
                if self.cfg.IMPROMPT.gvl==-1:
                    if self.cfg.IMPROMPT.embed_768_to_lang>=0:
                        if self.cfg.IMPROMPT.map_module == 'mlp':
                            visual_embeding_fcto768=self.fcto768(visual_embeding_flatten.contiguous().view(-1,49*256))
                        elif self.cfg.IMPROMPT.map_module == 'conv':
                            visual_embeding_fcto768=self.fcto768(reference_embeding)
                        language_dict_features['embedded'][:,self.cfg.IMPROMPT.embed_768_to_lang+1,:]=visual_embeding_fcto768
                        language_dict_features['hidden'][:, self.cfg.IMPROMPT.embed_768_to_lang+1, :] = visual_embeding_fcto768
                # visual_embeding_flatten = visual_embeding_flatten.permute(0, 2, 1)
                reference_feature=language_dict_features
                # reference_feature['hidden']=visual_embeding_flatten
                # reference_feature['embedded'] = visual_embeding_flatten
                #######################################################embed_768_to_lang end

                proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets,
                                                                             language_dict_features, positive_map,
                                                                             captions, swint_feature_c4,reference_feature)
            else:
                proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets,
                                                                         language_dict_features, positive_map,
                                                                         captions, swint_feature_c4)
        #########################################################################################################START
        if self.cfg.IMPROMPT.gvl==10:#wuyongjian:here we apply _proposals_ to get bboxes and refine the prediction
            langdict_sim=cos_similarity(language_dict_features['embedded'][0,:40,:])
            langdict_simh = cos_similarity(language_dict_features['hidden'][0,:,:])
            sim_2lang=cos_similarity2(language_dict_features['embedded'][0,:40,:],language_dict_features['hidden'][0,:40,:])
            # fused_visual_features = visual_features
            SIM_THRE=0.9
            boxes=proposals
            def get_box_feature_i_from_visual_feature(boxes, i, fused_visual_features):
                the_first_bbox_loc = boxes[0].extra_fields['loc'][i]
                the_first_bbox_stage = boxes[0].extra_fields['stage'][i]
                the_first_bbox_stagefeature = fused_visual_features[int(the_first_bbox_stage)]
                stage_size_h = the_first_bbox_stagefeature.shape[-1]
                the_first_bbox_feature = the_first_bbox_stagefeature[:, :, int(the_first_bbox_loc) // int(stage_size_h),
                                         int(the_first_bbox_loc) % int(stage_size_h)]
                return the_first_bbox_feature
            def refine_boxes(boxes0):
                boxes = boxes0.copy()
                boxes_len = len(boxes[0].extra_fields['loc'])
                THE_LAST_LOC = -1
                keep = []
                for i in range(boxes_len):
                    this_loc = int(boxes[0].extra_fields['loc'][i])
                    if THE_LAST_LOC != this_loc:
                        keep.append(i)
                        THE_LAST_LOC = this_loc
                boxes[0] = boxes[0][keep]
                return boxes
            rboxes = refine_boxes(boxes)
            rboxes_len = len(rboxes[0].extra_fields['loc'])
            b_features = torch.zeros((rboxes_len, 256))
            for i in range(rboxes_len):
                b_features[i, :] = get_box_feature_i_from_visual_feature(rboxes, i, fused_visual_features)
            import numpy
            target = 0
            # if not os.path.exists('example_boxori{}.npy'.format(target)):
            #     with open('example_boxori{}.npy'.format(target), 'wb') as f:
            #         numpy.save(f,b_features[target,:])
            #         example_box_feature=None
            # else:
            #     with open('example_box1.npy', 'rb') as f:
            #         example_box_feature=numpy.load(f)
            # if example_box_feature is not None:
            #     b_features[0,:]=torch.tensor(example_box_feature)
            b_similary = torch.matmul(b_features, b_features.transpose(-1, -2)).numpy()
            b_similary_to_range01=b_similary.copy()
            for i in range(b_similary.shape[0]):
                for j in range(b_similary.shape[1]):
                    b_similary_to_range01[i,j]/=numpy.sqrt(b_similary[i,i])
                    b_similary_to_range01[i, j] /= numpy.sqrt(b_similary[j, j])
            # b_similary_to_range01 = (b_similary - numpy.min(b_similary)) / (
            #             numpy.max(b_similary) - numpy.min(b_similary))
            VISUAL_SIM=True
            if VISUAL_SIM:
                import matplotlib.pyplot as plt
                plt.imshow(images.tensors[0].permute(1, 2, 0).cpu());
                bb = rboxes[0].bbox
                pred_classes=rboxes[0].extra_fields['labels'].cpu()
                for i in range(bb.shape[0])[:]:
                    bb2plot = bb[i, :]
                    x1, y1, x2, y2 = bb2plot.cpu()
                    if i!=target:
                        COLOR=[0, 1-b_similary_to_range01[target,i],b_similary_to_range01[target,i]]
                        COLOR = [2 - int(pred_classes[i]), int(pred_classes[i])-1,0]
                    else:
                        COLOR=[1,1,1]
                    if b_similary_to_range01[target, i]>SIM_THRE:
                        plt.text(x1, y1, '{}_{:.3f}'.format(i,b_similary_to_range01[target, i]))
                        linestyle='-'
                    else:
                        linestyle=':'
                    plt.gca().add_patch(
                        plt.Rectangle(xy=(x1, y1), width=(x2 - x1), height=(y2 - y1), edgecolor=COLOR, fill=False,
                                      linewidth=1,linestyle=linestyle))
                plt.show()
                # ref49 = reference_embeding[4].view((256, 49)).permute(1, 0).cpu()
                # ref10000 = reference_embeding[0].view((256,10000)).permute(1, 0).cpu()
                # wantsim = cos_similarity2(b_features, ref49)
        ##################################################################################################END
        if self.roi_heads:
            if self.cfg.MODEL.ROI_MASK_HEAD.PREDICTOR.startswith("VL"):
                if self.training:
                    # "Only support VL mask head right now!!"
                    assert len(targets) == 1 and len(targets[0]) == len(
                        positive_map), "shape match assert for mask head!!"
                    # Not necessary but as a safe guard:
                    # use the binary 0/1 positive map to replace the normalized positive map
                    targets[0].add_field("positive_map", positive_map)
            # TODO: make sure that this use of language_dict_features is correct!! Its content should be changed in self.rpn
            if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                x, result, detector_losses = self.roi_heads(
                    fused_visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
            else:
                x, result, detector_losses = self.roi_heads(
                    visual_features, proposals, targets,
                    language_dict_features=language_dict_features,
                    positive_map_label_to_token=positive_map if not self.training else None
                )
        else:
            # RPN-only models don't have roi_heads
            x = visual_features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

    def _forward_language_parallel(self, captions=None, targets=None,
                                   device=None, positive_map=None):
        ktype = self.cfg.GLIPKNOW.KNOWLEDGE_TYPE

        def _construct_captions_from_class_names(class_names):
            captions = []
            for c in class_names:
                try:
                    info = self.class_name_to_knowledge[c]
                    cap = info['clean_name']

                    # combine wiki and gpt3 knowledge
                    if self.cfg.GLIPKNOW.WIKI_AND_GPT3:
                        ktype = 'def_wiki'
                        know_seq = info[ktype]

                        ktype = 'gpt3'
                        if ktype == 'gpt3' or type(info[ktype]) == list:
                            know_seq += ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM]])

                        cap += ': ' + know_seq

                    # only one knoweldge source is used
                    else:
                        if ktype and ktype in info and info[ktype]:
                            if ktype == 'gpt3' or type(info[ktype]) == list:
                                know_seq = ' '.join([seq for seq in info[ktype][:self.cfg.GLIPKNOW.GPT3_NUM]])
                            else:
                                know_seq = info[ktype]
                            cap += ': ' + know_seq
                except:
                    cap = c
                    print(f'cap {cap}, c {c}')

                captions.append(cap)
            return captions

        if self.training:
            assert captions is None
            assert targets is not None

            max_classes_per_batch = self.cfg.GLIPKNOW.MAX_NUM_CLASSES_PER_BATCH_TRAIN
            if max_classes_per_batch >= len(self.class_name_list):
                shuffled_class_names = self.class_name_list.copy()
                random.shuffle(shuffled_class_names)
                if max_classes_per_batch > len(shuffled_class_names):
                    shuffled_class_names.extend(shuffled_class_names[:max_classes_per_batch
                                                                      - len(shuffled_class_names)])
                    random.shuffle(shuffled_class_names)
            else:
                label_list = []
                label_to_idx = {}
                for target_per_im in targets:
                    labels_per_im = target_per_im.get_field('label_names')
                    for label in labels_per_im:
                        if label not in label_to_idx:
                            label_to_idx[label] = len(label_list)
                            label_list.append(label)

                label_list = label_list[:max_classes_per_batch]
                if len(label_list) < max_classes_per_batch:
                    all_neg_classes = [c for c in self.class_name_list if c not
                                       in label_to_idx]
                    neg_label_list = random.sample(all_neg_classes,
                                                   max_classes_per_batch - len(label_list))
                    label_list.extend(neg_label_list)
                random.shuffle(label_list)
                shuffled_class_names = label_list

            label_to_shuffled_idx = {l: i for i, l in
                                     enumerate(shuffled_class_names)}
            total_boxes = sum(len(t) for t in targets)
            positive_map = torch.zeros((total_boxes, max_classes_per_batch + 1),
                                       device=device)
            offset = 0
            for target_per_im in targets:
                labels_per_im = target_per_im.get_field('label_names')
                for label in labels_per_im:
                    j = label_to_shuffled_idx.get(label, -1)
                    if j >= 0:
                        positive_map[offset, j] = 1
                    offset += 1
            captions = _construct_captions_from_class_names(shuffled_class_names)
            captions.append('')  # onobj at the end, onedet/modeling/rpn/loss.py:719
            batch_size = len(targets)

        else:
            assert captions is not None
            batch_size = 1
            assert len(captions) == 1
            class_names = captions[0]
            max_classes_per_batch = len(class_names)
            captions = _construct_captions_from_class_names(class_names)
            captions.append('')  # onobj at the end, onedet/modeling/rpn/loss.py:719

        tokenized = self.tokenizer.batch_encode_plus(captions,
                                                     max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                     padding="longest",
                                                     return_special_tokens_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True).to(device)
        assert not self.use_mlm_loss
        tokenizer_input = {"input_ids": tokenized.input_ids,
                           "attention_mask": tokenized.attention_mask}

        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)
        else:
            language_dict_features = self.language_backbone(tokenizer_input)

        assert not self.cfg.DATASETS.ONE_HOT
        assert not self.cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL

        agg_type = self.cfg.GLIPKNOW.LAN_FEATURE_AGG_TYPE
        agg_feats = language_dict_features['hidden']
        agg_emb = language_dict_features['embedded']
        if agg_type == 'first':
            agg_feats = agg_feats[:, 0, :]
            agg_emb = agg_emb[:, 0, :]
        elif agg_type == 'mean':
            attn_mask = language_dict_features['masks']
            seq_len = attn_mask.sum(-1).unsqueeze(-1).float()
            agg_feats = agg_feats * attn_mask.unsqueeze(-1).float()
            agg_feats = agg_feats.sum(1) / seq_len
            agg_emb = agg_emb * attn_mask.unsqueeze(-1).float()
            agg_emb = agg_emb.sum(1) / seq_len
        else:
            raise ValueError('not supported GLIPKNOW.LAN_FEATURE_AGG_TYPE: {}'.format(agg_type))

        expanded_features = agg_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        expanded_embedding = agg_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        lang_dict = {}
        lang_dict["mlm_labels"] = None
        lang_dict["aggregate"] = None
        lang_dict["embedded"] = expanded_embedding
        lang_dict['hidden'] = expanded_features
        lang_dict["masks"] = torch.ones((batch_size, max_classes_per_batch + 1),
                                        device=device, dtype=language_dict_features['masks'].dtype)
        # in GLIP setting, the token at the end of seqence is usually [PAD], and is masked out
        # if [noobj] is not masked out, the loss sum is very big, as most
        # anchors are matched to [noobj]
        lang_dict["masks"][:, -1] = 0
        return lang_dict, positive_map
