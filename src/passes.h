#pragma once

#include <d3d12.h>
#include <cstdint>
#include <string>
#include <vector>

// Root constants of prepare.hlsl, in the order its cbuffer declares them.
struct prepare_constants
{
    float size[4];
    float depth_params[4];
    float inv_proj[16];
    float cam_to_world[12];
    float prev_view_proj[12];
    float jitter_delta[4];
};
static_assert(sizeof(prepare_constants) == 52 * 4, "prepare.hlsl expects 52 root constants");

struct game_texture
{
    ID3D12Resource *resource = nullptr;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN; // the format the game reads it with
};

// The compute passes around DLSS and the textures they share. Everything is
// recorded with plain D3D12 into the game's command list.
class passes
{
public:
    bool init(ID3D12Device *device);
    void shutdown();

    // Rebuilds the work textures when a size or the color format changes. The
    // old ones stay alive until `frame` is far enough behind.
    bool resize(uint32_t width, uint32_t height, uint32_t output_width, uint32_t output_height,
                DXGI_FORMAT color_format, uint64_t frame);
    void collect(uint64_t frame);

    // Fills color, motion and depth and leaves them readable by DLSS.
    void prepare(ID3D12GraphicsCommandList *list, const game_texture &color, const game_texture &motion,
                 const game_texture &depth, const prepare_constants &constants);

    // Sharpens if asked, then blurs along the motion vectors if asked, and
    // hands the result over. At DLAA it is copied into
    // the game's TAA target. When upscaling, it goes into presented() for the
    // pass that scales the frame to the screen, and a render-size copy into
    // the TAA target for everything else that reads it. Every work texture is
    // then back in the state prepare() expects.
    // blur is the shutter, 0 skips the pass; blur_limit the longest blur as a
    // fraction of the screen height.
    void finish(ID3D12GraphicsCommandList *list, float sharpness, bool hdr, ID3D12Resource *taa_target, float blur,
                float blur_limit, uint64_t frame);
    // For when DLSS did not run after prepare(): only the states go back.
    void abandon(ID3D12GraphicsCommandList *list);

    // Resamples presented() into a game texture in the unordered access state,
    // at its own size and format. This is how the frame reaches the screen
    // when the game's own FSR 1 upscales.
    void deliver(ID3D12GraphicsCommandList *list, ID3D12Resource *target, DXGI_FORMAT format, uint32_t width,
                 uint32_t height);

    bool upscaling() const { return _output_width != _width || _output_height != _height; }

    ID3D12Resource *color() const { return _color; }
    ID3D12Resource *motion() const { return _motion; }
    ID3D12Resource *depth() const { return _depth; }
    ID3D12Resource *output() const { return _output; }
    // The upscaled frame, always readable by any shader between frames.
    ID3D12Resource *presented() const { return _presented; }
    DXGI_FORMAT color_format() const { return _color_format; }
    const std::string &error() const { return _error; }

private:
    ID3D12Resource *make_texture(DXGI_FORMAT format, uint32_t width, uint32_t height, D3D12_RESOURCE_STATES state);
    void release_textures(uint64_t frame);
    D3D12_GPU_DESCRIPTOR_HANDLE next_slot(D3D12_CPU_DESCRIPTOR_HANDLE &cpu);
    D3D12_CPU_DESCRIPTOR_HANDLE offset(D3D12_CPU_DESCRIPTOR_HANDLE base, uint32_t index) const;
    // Motion blur of `source` (output size, readable) into _blurred, using the
    // motion vectors in _motion.
    void blur(ID3D12GraphicsCommandList *list, ID3D12Resource *source, float strength, float max_length,
              uint64_t frame);
    void write_srv(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource, DXGI_FORMAT format);
    void write_uav(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource, DXGI_FORMAT format);
    void dispatch(ID3D12GraphicsCommandList *list, ID3D12PipelineState *pipeline, const void *constants,
                  uint32_t constant_count, ID3D12Resource *source, ID3D12Resource *target, DXGI_FORMAT target_format,
                  uint32_t width, uint32_t height);

    ID3D12Device *_device = nullptr;
    ID3D12RootSignature *_root = nullptr;
    ID3D12PipelineState *_prepare = nullptr;
    ID3D12PipelineState *_rcas = nullptr;
    ID3D12PipelineState *_downsample = nullptr;
    ID3D12PipelineState *_motionblur = nullptr;
    ID3D12DescriptorHeap *_heap = nullptr;
    uint32_t _increment = 0;
    uint32_t _next_slot = 0;

    // Render size: what the game hands over.
    ID3D12Resource *_color = nullptr, *_motion = nullptr, *_depth = nullptr, *_downsampled = nullptr;
    // Output size: what DLSS makes.
    ID3D12Resource *_output = nullptr, *_sharpened = nullptr, *_blurred = nullptr, *_presented = nullptr;
    uint32_t _width = 0, _height = 0, _output_width = 0, _output_height = 0;
    DXGI_FORMAT _color_format = DXGI_FORMAT_UNKNOWN;

    struct retired
    {
        ID3D12Resource *resource;
        uint64_t frame;
    };
    std::vector<retired> _retired;
    std::string _error;
};
