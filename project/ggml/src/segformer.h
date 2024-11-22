#ifndef __SEGFORMER__H__
#define __SEGFORMER__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"

#include <vector>

/*
 ConvModule(
  (conv): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (bn): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (activate): ReLU()
) */

struct ConvModule {
    // network hparams
    
    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = 768 * 4;
        conv.out_channels = 768;
        conv.kernel_size = {1, 1};
        conv.stride = { 1, 1 };
        conv.padding = { 0, 0 };
        // conv.dilation = { 1, 1 };
        // conv.is_depthwise = false;
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);

        bn.num_features = 768;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);
        x = ggml_relu(ctx, x);

        return x;
    }


};

/*
 MLP(
  (proj): Linear(in_features=512, out_features=768, bias=True)
) */

struct MLP {
    int input_dim = 512;
    // network hparams
    
    // network params
    struct Linear proj;
    // self.proj = nn.Linear(input_dim, 768)

    void create_weight_tensors(struct ggml_context* ctx) {
        proj.in_features = input_dim;
        proj.out_features = 768;
        proj.has_bias = true;
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // B, C, H, W = x.size()
        // x = x.flatten(2).transpose(1, 2)
        // x = self.proj(x)
        // return x.permute(0, 2, 1)
        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        x = ggml_reshape_3d(ctx, x, W*H, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        x = proj.forward(ctx, x);

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        return x;
    }
};

/*
 SegFormerHead(
  (conv_seg): Conv2d(128, 150, kernel_size=(1, 1), stride=(1, 1))
  (linear_c1): MLP(
    (proj): Linear(in_features=64, out_features=768, bias=True)
  )
  (linear_c2): MLP(
    (proj): Linear(in_features=128, out_features=768, bias=True)
  )
  (linear_c3): MLP(
    (proj): Linear(in_features=320, out_features=768, bias=True)
  )
  (linear_c4): MLP(
    (proj): Linear(in_features=512, out_features=768, bias=True)
  )
  (linear_fuse): ConvModule(
    (conv): Conv2d(3072, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activate): ReLU()
  )
  (linear_pred): Conv2d(768, 150, kernel_size=(1, 1), stride=(1, 1))
) */

struct SegFormerHead {
    // network hparams
    int num_classes = 150;

    // network params
    struct Conv2d conv_seg;
    struct MLP linear_c1;
    struct MLP linear_c2;
    struct MLP linear_c3;
    struct MLP linear_c4;
    struct ConvModule linear_fuse;
    struct Conv2d linear_pred;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.conv_seg = nn.Conv2d(128, self.num_classes, kernel_size=1)
        conv_seg.in_channels = 128;
        conv_seg.out_channels = num_classes;
        conv_seg.kernel_size = {1, 1};
        conv_seg.stride = { 1, 1 };
        conv_seg.padding = { 0, 0 };
        // conv_seg.dilation = { 1, 1 };
        // conv_seg.is_depthwise = false;
        conv_seg.has_bias = true;
        conv_seg.create_weight_tensors(ctx);

        // 64, 128, 320, 512
        linear_c1.input_dim = 64;
        linear_c1.create_weight_tensors(ctx);
        linear_c2.input_dim = 128;
        linear_c2.create_weight_tensors(ctx);
        linear_c3.input_dim = 320;
        linear_c3.create_weight_tensors(ctx);
        linear_c4.input_dim = 512;
        linear_c4.create_weight_tensors(ctx);

        // self.linear_fuse = ConvModule(in_channels=768 * 4, out_channels=768)
        linear_fuse.create_weight_tensors(ctx);

        // self.linear_pred = nn.Conv2d(768, self.num_classes, kernel_size=1)
        linear_pred.in_channels = 768;
        linear_pred.out_channels = num_classes;
        linear_pred.kernel_size = {1, 1};
        linear_pred.stride = { 1, 1 };
        linear_pred.padding = { 0, 0 };
        // linear_pred.dilation = { 1, 1 };
        // linear_pred.is_depthwise = false;
        linear_pred.has_bias = true;
        linear_pred.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv_seg.");
        conv_seg.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c1.");
        linear_c1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c2.");
        linear_c2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c3.");
        linear_c3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_c4.");
        linear_c4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "linear_fuse.");
        linear_fuse.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "linear_pred.");
        linear_pred.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, std::vector<ggml_tensor_t *> xlist) {
        // # inputs is tuple: len = 4
        // #     tensor [item] size: [1, 64, 240, 320], min: -4.25842, max: 4.218358, mean: 0.014021
        // #     tensor [item] size: [1, 128, 120, 160], min: -6.090078, max: 4.901278, mean: 0.02357
        // #     tensor [item] size: [1, 320, 60, 80], min: -5.592515, max: 4.761344, mean: -0.002071
        // #     tensor [item] size: [1, 512, 30, 40], min: -6.624208, max: 8.50036, mean: 0.025605
        ggml_tensor_t *x1 = xlist[0];
        ggml_tensor_t *x2 = xlist[1];
        ggml_tensor_t *x3 = xlist[2];
        ggml_tensor_t *x4 = xlist[3];

        int W1 = (int)x1->ne[0];
        int H1 = (int)x1->ne[1];
        int C1 = (int)x1->ne[2];
        int B1 = (int)x1->ne[3];

        int W4 = (int)x4->ne[0];
        int H4 = (int)x4->ne[1];
        int C4 = (int)x4->ne[2];
        int B4 = (int)x4->ne[3];

        // x = inputs  # len=4, 1/4,1/8,1/16,1/32
        // # c1, c2, c3, c4 = x # onnx not happy
        // c1 = x[0]
        // c2 = x[1]
        // c3 = x[2]
        // c4 = x[3]

        // ############## MLP decoder on C1-C4 ###########
        // B1, C1, H1, W1 = c1.size()
        // B2, C2, H2, W2 = c2.size()
        // B3, C3, H3, W3 = c3.size()
        // B4, C4, H4, W4 = c4.size()

        // _c4 = self.linear_c4(c4).reshape(B4, -1, H4, W4)
        // _c4 = F.interpolate(_c4, size=(H1, W1), mode="bilinear", align_corners=False)

        // x4
        {
            x4 = linear_c4.forward(ctx, x4);
            C4 = ggml_nelements(x4)/B4/H4/W4; // update C4
            x4 = ggml_reshape_4d(ctx, x4, B4, C4, H4, W4);

            x4 = ggml_interpolate(ctx, x4, 0, W1); 
            x4 = ggml_interpolate(ctx, x4, 1, H1); 
        }

        // _c3 = self.linear_c3(c3).reshape(B4, -1, H3, W3)
        // _c3 = F.interpolate(_c3, size=(H1, W1), mode="bilinear", align_corners=False)

        // x3
        {
            int W3 = (int)x3->ne[0];
            int H3 = (int)x3->ne[1];
            int C3 = (int)x3->ne[2];
            int B3 = (int)x3->ne[3];

            x3 = linear_c3.forward(ctx, x3);
            C3 = ggml_nelements(x3)/B4/H3/W3; // update C4
            x3 = ggml_reshape_4d(ctx, x3, B4, C3, H3, W3);

            x3 = ggml_interpolate(ctx, x3, 0, W1); 
            x3 = ggml_interpolate(ctx, x3, 1, H1); 
        }

        // _c2 = self.linear_c2(c2).reshape(B4, -1, H2, W2)
        // _c2 = F.interpolate(_c2, size=(H1, W1), mode="bilinear", align_corners=False)
        // x2
        {
            int W2 = (int)x2->ne[0];
            int H2 = (int)x2->ne[1];
            int C2 = (int)x2->ne[2];
            int B2 = (int)x2->ne[3];

            x2 = linear_c2.forward(ctx, x2);
            C2 = ggml_nelements(x2)/B4/H2/W2; // update C4
            x2 = ggml_reshape_4d(ctx, x2, B4, C2, H2, W2);

            x2 = ggml_interpolate(ctx, x2, 0, W1); 
            x2 = ggml_interpolate(ctx, x2, 1, H1); 
        }

        // _c1 = self.linear_c1(c1).reshape(B4, -1, H1, W1)
        // x1
        {
            x1 = linear_c1.forward(ctx, x1);
            C1 = ggml_nelements(x1)/B4/H1/W1; // update C4
            x1 = ggml_reshape_4d(ctx, x1, B4, C1, H1, W1);
            // x1 = ggml_interpolate(ctx, x1, 0, W1); 
            // x1 = ggml_interpolate(ctx, x1, 1, H1); 
        }
        ggml_tensor_t *x1234 = ggml_concat(ctx, x4, x3, 2/*dim on channels*/);
        x1234 = ggml_concat(ctx, x1234, x2, 2/*dim on channels*/);
        x1234 = ggml_concat(ctx, x1234, x1, 2/*dim on channels*/);

        ggml_tensor_t* x = linear_fuse.forward(ctx, x1234);
        x = linear_pred.forward(ctx, x);        

        return x;
    }
};

/*
 DWConv(
  (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
) */

struct DWConv {
    int dim = 768;

    // network params
    struct Conv2d dwconv;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        dwconv.in_channels = dim;
        dwconv.out_channels = dim;
        dwconv.kernel_size = {3, 3};
        dwconv.stride = { 1, 1 };
        dwconv.padding = { 1, 1 };
        // dwconv.dilation = { 1, 1 };
        dwconv.is_depthwise = true;
        dwconv.has_bias = true;
        dwconv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "dwconv.");
        dwconv.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int H, int W) {
        // B, N, C = x.shape
        // x = x.transpose(1, 2).view(B, C, H, W)
        // x = self.dwconv(x)
        // x = x.flatten(2).transpose(1, 2)
        // return x
        int B = (int)x->ne[2];
        // int N = (int)x->ne[1];
        int C = (int)x->ne[0];
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        x = ggml_reshape_4d(ctx, x, W, H, C, B);
        x = dwconv.forward(ctx, x);
        W = (int)x->ne[0];
        H = (int)x->ne[1];
        C = (int)x->ne[2];
        B = (int)x->ne[3];
        x = ggml_reshape_3d(ctx, x, W*H, C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        return x;
    }
};

/*
 BlockMlp(
  (fc1): Linear(in_features=512, out_features=2048, bias=True)
  (dwconv): DWConv(
    (dwconv): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
  )
  (act): GELU(approximate='none')
  (fc2): Linear(in_features=2048, out_features=512, bias=True)
) */

struct BlockMlp {
    int in_features;
    int hidden_features;

    // network params
    struct Linear fc1;
    struct DWConv dwconv;
    struct Linear fc2;

    void create_weight_tensors(struct ggml_context* ctx) {
        fc1.in_features = in_features;
        fc1.out_features = hidden_features;
        fc1.create_weight_tensors(ctx);

        dwconv.dim = hidden_features;
        dwconv.create_weight_tensors(ctx);

        fc2.in_features = hidden_features;
        fc2.out_features = in_features;
        fc2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "fc1.");
        fc1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dwconv.");
        dwconv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fc2.");
        fc2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int H, int W) {
        // x = self.fc1(x)
        // x = self.dwconv(x, H, W)
        // x = self.act(x)
        // x = self.fc2(x)
        // return x

        x = fc1.forward(ctx, x);
        x = dwconv.forward(ctx, x, H, W);
        x = ggml_gelu(ctx, x);
        x = fc2.forward(ctx, x);
    	return x;
    }
};

/*
 Attention(
  (q): Linear(in_features=512, out_features=512, bias=True)
  (kv): Linear(in_features=512, out_features=1024, bias=True)
  (proj): Linear(in_features=512, out_features=512, bias=True)
  (sr): Identity()
  (norm): Identity()
) */

struct Attention {
    // network hparams
    int dim = 512;
    int num_heads = 8;
    int sr_ratio = 1;

    float scale = 0.125;

    // network params
    struct Linear q;
    struct Linear kv;
    struct Linear proj;

    struct Conv2d sr;
    struct LayerNorm norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        q.in_features = dim;
        q.out_features = dim;
        q.has_bias = true;
        q.create_weight_tensors(ctx);

        kv.in_features = dim;
        kv.out_features = 2*dim;
        kv.has_bias = true;
        kv.create_weight_tensors(ctx);

        proj.in_features = dim;
        proj.out_features = dim;
        proj.has_bias = false;
        proj.create_weight_tensors(ctx);

        if (sr_ratio > 1) {
            sr.in_channels = dim;
            sr.out_channels = dim;

            // Fixed defaults ...
            sr.kernel_size = { sr_ratio, sr_ratio };
            sr.stride = { sr_ratio, sr_ratio };
            sr.padding = { 0, 0 };
            sr.dilation = { 1, 1 };
            sr.is_depthwise = false;
            sr.has_bias = true;
            sr.create_weight_tensors(ctx);

            norm.normalized_shape = dim;
            norm.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "q.");
        q.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "kv.");
        kv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);

        if (sr_ratio > 1) {
            snprintf(s, sizeof(s), "%s%s", prefix, "sr.");
            sr.setup_weight_names(s);

            snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
            norm.setup_weight_names(s);
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int H, int W) {
    	// please implement forward by your self, please !!!
        // xxxx_debug
    	return x;
    }
};

struct Block {
    int dim;
    int num_heads;
    int sr_ratio = 1;

    // network params
    struct LayerNorm norm1;
    struct Attention attn;
    struct LayerNorm norm2;
    struct BlockMlp mlp;

    void create_weight_tensors(struct ggml_context* ctx) {
        norm1.normalized_shape = dim;
        norm1.create_weight_tensors(ctx);

        attn.dim = dim;
        attn.num_heads = num_heads;
        attn,sr_ratio = sr_ratio;
        attn.create_weight_tensors(ctx);

        norm2.normalized_shape = dim;
        norm2.create_weight_tensors(ctx);

        mlp.in_features = dim;
        mlp.hidden_features = dim * 4;
        mlp.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "attn.");
        attn.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int H, int W) {
        // x = x + self.attn(self.norm1(x), H, W) 
        // x = x + self.mlp(self.norm2(x), H, W)
        // return x

        ggml_tensor_t *t = norm1.forward(ctx, x);
        t = attn.forward(ctx, x, H, W);
        x = ggml_add(ctx, x, t);

        t = norm2.forward(ctx, x);
        t = mlp.forward(ctx, t, H, W);
        x = ggml_add(ctx, x, t);

    	return x;
    }
};

/*
 OverlapPatchEmbed(
  (proj): Conv2d(320, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
) */

struct OverlapPatchEmbed {
    // network hparams
    int in_chans = 3;
    int embed_dim = 512;
    int stride = 4;
    int patch_size = 7;

    struct Conv2d proj;
    struct LayerNorm norm;

    void create_weight_tensors(struct ggml_context* ctx) {
        proj.in_channels = in_chans;
        proj.out_channels = embed_dim;
        proj.kernel_size = { patch_size, patch_size };
        proj.stride = { stride, stride };
        proj.padding = { patch_size/2, patch_size/2 };
        // proj.dilation = { 1, 1 };
        proj.is_depthwise = false;
        proj.has_bias = true;
        proj.create_weight_tensors(ctx);

        norm.normalized_shape = embed_dim;
        norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int *H, int *W) {
        // x = self.proj(x)
        // proj_out = x
        // # tensor [x] size: [1, 64, 240, 320], min: -8.812546, max: 10.21946, mean: -0.002233
        // # tensor [x] size: [1, 128, 120, 160], min: -10.213951, max: 6.449842, mean: -0.02391

        // x = x.flatten(2).transpose(1, 2)
        // x = self.norm(x)

        // return (x, proj_out)
        x = proj.forward(ctx, x);
        *W = (int)x->ne[0];
        *H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];
        x = ggml_reshape_3d(ctx, x, (*W)*(*H), C, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        x = norm.forward(ctx, x);

        return x;
    }
};


struct VisionTransformer {
    // network hparams
    int embed_dims[4] = {64, 128, 320, 512};
    int num_heads[4] = {1, 2, 5, 8};
    int depths[4] = {3, 8, 27, 3};
    int sr_ratios[4] = {8, 4, 2, 1};

    // network params
    struct OverlapPatchEmbed patch_embed1;
    struct OverlapPatchEmbed patch_embed2;
    struct OverlapPatchEmbed patch_embed3;
    struct OverlapPatchEmbed patch_embed4;

    // depth [3, 8, 27, 3]
    struct Block block1[3];
    struct LayerNorm norm1;

    struct Block block2[8];
    struct LayerNorm norm2;

    struct Block block3[27];
    struct LayerNorm norm3;

    struct Block block4[3];
    struct LayerNorm norm4;

    void create_weight_tensors(struct ggml_context* ctx) {
        // embed_dims=[64, 128, 320, 512]
        // self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        // self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        // self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        // self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        patch_embed1.in_chans = 3;
        patch_embed1.embed_dim = 64;
        patch_embed1.stride = 4;
        patch_embed1.patch_size = 7;
        patch_embed1.create_weight_tensors(ctx);

        patch_embed2.in_chans = 64;
        patch_embed2.embed_dim = 128;
        patch_embed2.stride = 2;
        patch_embed2.patch_size = 3;
        patch_embed2.create_weight_tensors(ctx);

        patch_embed3.in_chans = 128;
        patch_embed3.embed_dim = 320;
        patch_embed3.stride = 2;
        patch_embed3.patch_size = 3;
        patch_embed3.create_weight_tensors(ctx);

        patch_embed4.in_chans = 320;
        patch_embed4.embed_dim = 512;
        patch_embed4.stride = 2;
        patch_embed4.patch_size = 3;
        patch_embed4.create_weight_tensors(ctx);

        for (int i = 0; i < depths[0]; i++) {
            block1[i].dim = embed_dims[0];
            block1[i].num_heads = num_heads[0];
            block1[i].sr_ratio = sr_ratios[0];

            block1[i].create_weight_tensors(ctx);
        }
        norm1.normalized_shape = embed_dims[0];
        norm1.create_weight_tensors(ctx);

        for (int i = 0; i < depths[1]; i++) {
            block2[i].dim = embed_dims[1];
            block2[i].num_heads = num_heads[1];
            block2[i].sr_ratio = sr_ratios[1];

            block2[i].create_weight_tensors(ctx);
        }
        // block2.create_weight_tensors(ctx);

        norm2.normalized_shape = embed_dims[1];
        norm2.create_weight_tensors(ctx);

        for (int i = 0; i < depths[2]; i++) {
            block3[i].dim = embed_dims[2];
            block3[i].num_heads = num_heads[2];
            block3[i].sr_ratio = sr_ratios[2];

            block3[i].create_weight_tensors(ctx);
        }
        // block3.create_weight_tensors(ctx);
        norm3.normalized_shape = embed_dims[2];
        norm3.create_weight_tensors(ctx);

        for (int i = 0; i < depths[3]; i++) {
            block4[i].dim = embed_dims[3];
            block4[i].num_heads = num_heads[3];
            block4[i].sr_ratio = sr_ratios[3];

            block4[i].create_weight_tensors(ctx);
        }
        // block4.create_weight_tensors(ctx);
        norm4.normalized_shape = embed_dims[3];
        norm4.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed1.");
        patch_embed1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed2.");
        patch_embed2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed3.");
        patch_embed3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "patch_embed4.");
        patch_embed4.setup_weight_names(s);

        for (int i = 0; i < depths[0]; i++) {
            // snprintf(s, sizeof(s), "%s%s", prefix, "block1.0.");
            snprintf(s, sizeof(s), "%sblock1.%d.", prefix, i);
            block1[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);

        for (int i = 0; i < depths[1]; i++) {
            snprintf(s, sizeof(s), "%sblock2.%d.", prefix, i);
            block2[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);

        for (int i = 0; i < depths[2]; i++) {
            snprintf(s, sizeof(s), "%sblock3.%d.", prefix, i);
            block3[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm3.setup_weight_names(s);

        for (int i = 0; i < depths[3]; i++) {
            snprintf(s, sizeof(s), "%sblock4.%d.", prefix, i);
            block4[i].setup_weight_names(s);
        }
        snprintf(s, sizeof(s), "%s%s", prefix, "norm4.");
        norm4.setup_weight_names(s);
    }

    std::vector<struct ggml_tensor*> forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        std::vector<struct ggml_tensor*> xlist;

        // B = x.shape[0]

        // # stage 1
        int B = (int)x->ne[3];
        int C = (int)x->ne[2];
        int H = (int)x->ne[1];
        int W = (int)x->ne[0];

        // x, x_proj_out = self.patch_embed1(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block1):
        //     x = blk(x, H, W)
        // x = self.norm1(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x1 = x
        x = patch_embed1.forward(ctx, x, &H, &W);
        for (int i = 0; i < depths[0]; i++) {
            x = block1[i].forward(ctx, x, H, W);
        }
        x = norm1.forward(ctx, x);
        C = ggml_nelements(x)/B/H/W;
        x = ggml_reshape_4d(ctx, x, C, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // # stage 2
        // x, x_proj_out = self.patch_embed2(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block2):
        //     x = blk(x, H, W)
        // x = self.norm2(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x2 = x
        x = patch_embed2.forward(ctx, x, &H, &W);
        for (int i = 0; i < depths[1]; i++) {
            x = block2[i].forward(ctx, x, H, W);
        }
        x = norm2.forward(ctx, x);
        C = ggml_nelements(x)/B/H/W;
        x = ggml_reshape_4d(ctx, x, C, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // # stage 3
        // x, x_proj_out = self.patch_embed3(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block3):
        //     x = blk(x, H, W)
        // x = self.norm3(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x3 = x
        x = patch_embed3.forward(ctx, x, &H, &W);
        for (int i = 0; i < depths[2]; i++) {
            x = block3[i].forward(ctx, x, H, W);
        }
        x = norm3.forward(ctx, x);
        C = ggml_nelements(x)/B/H/W;
        x = ggml_reshape_4d(ctx, x, C, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // # stage 4
        // x, x_proj_out = self.patch_embed4(x)
        // _, _, H, W = x_proj_out.shape
        // for i, blk in enumerate(self.block4):
        //     x = blk(x, H, W)
        // x = self.norm4(x)
        // x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        // x4 = x
        x = patch_embed4.forward(ctx, x, &H, &W);
        for (int i = 0; i < depths[3]; i++) {
            x = block4[i].forward(ctx, x, H, W);
        }
        x = norm4.forward(ctx, x);
        C = ggml_nelements(x)/B/H/W;
        x = ggml_reshape_4d(ctx, x, C, W, H, B);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)); // [C, W, H, B] -> [W, H, C, B]
        xlist.push_back(x);

        // return (x1, x2, x3, x4)
    	return xlist;
    }
};


struct SegmentModel : GGMLNetwork {
    // network hparams
    int MAX_H = 1024;
    int MAX_W = 2048;
    int MAX_TIMES = 4;
    int num_classes = 150;

    // network params
    struct Normalize normalize;
    struct VisionTransformer backbone;
    struct SegFormerHead decode_head;


    void create_weight_tensors(struct ggml_context* ctx) {
        backbone.create_weight_tensors(ctx);
        decode_head.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "backbone.");
        backbone.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "decode_head.");
        decode_head.setup_weight_names(s);
    }

    // B, C, H, W = x.shape
    // # x.size() -- ([1, 3, 960, 1280])
    // r_pad = (self.MAX_TIMES - (W % self.MAX_TIMES)) % self.MAX_TIMES
    // b_pad = (self.MAX_TIMES - (H % self.MAX_TIMES)) % self.MAX_TIMES
    // x = F.pad(x, (0, r_pad, 0, b_pad), mode="replicate")

    // x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    // f = self.backbone(x)
    // # f is tuple: len = 4
    // #     tensor [item] size: [1, 64, 240, 320], min: -4.25842, max: 4.218358, mean: 0.014021
    // #     tensor [item] size: [1, 128, 120, 160], min: -6.090078, max: 4.901278, mean: 0.02357
    // #     tensor [item] size: [1, 320, 60, 80], min: -5.592515, max: 4.761344, mean: -0.002071
    // #     tensor [item] size: [1, 512, 30, 40], min: -6.624208, max: 8.50036, mean: 0.025605

    // seg_logit = self.decode_head(f)
    // # inputs is tuple: len = 4
    // #     tensor [item] size: [1, 64, 240, 320], min: -4.25842, max: 4.218358, mean: 0.014021
    // #     tensor [item] size: [1, 128, 120, 160], min: -6.090078, max: 4.901278, mean: 0.02357
    // #     tensor [item] size: [1, 320, 60, 80], min: -5.592515, max: 4.761344, mean: -0.002071
    // #     tensor [item] size: [1, 512, 30, 40], min: -6.624208, max: 8.50036, mean: 0.025605

    // seg_logit = F.interpolate(seg_logit, size=(H,W), mode="bilinear", align_corners=False)
    // seg_logit = F.softmax(seg_logit, dim=1) # [1, 150, 960, 1280]

    // mask = seg_logit.argmax(dim=1).unsqueeze(0) # [1, 960, 1280] -> [1, 1, 960, 1280]

    // mask = mask[:, :, 0:H, 0:W]
    // mask = mask.clamp(0, self.num_classes).to(torch.float32)
    // # ADE20K class number is 150, to float32 is for onnx export
    // # tensor [mask] size: [1, 1, 960, 1280], min: 0.0, max: 25.0, mean: 3.283945        

    // return mask

    // struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_UNUSED(argc);
        ggml_tensor_t* x = argv[0];

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        int C = (int)x->ne[2];
        int B = (int)x->ne[3];

        int r_pad = (MAX_TIMES - (W % MAX_TIMES)) % MAX_TIMES;
        int b_pad = (MAX_TIMES - (H % MAX_TIMES)) % MAX_TIMES;

        x = ggml_replication_pad2d(ctx, x, 0, r_pad, 0, b_pad);

        x = normalize.forward(ctx, x);
        std::vector<ggml_tensor_t *>xlist = backbone.forward(ctx, x);
        ggml_tensor_t *seg_logit = decode_head.forward(ctx, xlist);

        seg_logit = ggml_interpolate(ctx, seg_logit, 1, H);  
        seg_logit = ggml_interpolate(ctx, seg_logit, 0, W); 
        seg_logit = ggml_soft_max(ctx, seg_logit);

        ggml_tensor_t *mask = ggml_argmax(ctx, seg_logit);
        mask = ggml_clamp(ctx, mask, 0.0, (float)num_classes);

        return mask;
    }
};

#endif // __SEGFORMER__H__
