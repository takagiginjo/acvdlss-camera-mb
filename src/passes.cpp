#include "passes.h"

#include <downsample.h>
#include <motionblur.h>
#include <prepare.h>
#include <rcas.h>

#include <cstdio>
#include <iterator>

// Six descriptors per dispatch: three inputs, three outputs. At most four
// dispatches a frame, so a slot comes round again after 16 frames, far more
// than the GPU ever trails behind.
static const uint32_t SLOT_SIZE = 6;
static const uint32_t SLOT_COUNT = 64;
// Frames a retired texture is kept for. The game queues at most three.
static const uint64_t RETIRE_FRAMES = 8;

static const D3D12_RESOURCE_STATES READABLE =
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE;

static D3D12_RESOURCE_BARRIER transition(ID3D12Resource *resource, D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

template <typename T>
static void release(T *&object)
{
    if (object != nullptr)
        object->Release();
    object = nullptr;
}

bool passes::init(ID3D12Device *device)
{
    _device = device;

    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 3;
    ranges[1].OffsetInDescriptorsFromTableStart = 3;

    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = sizeof(prepare_constants) / 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 2;
    desc.pParameters = params;
    desc.NumStaticSamplers = 1;
    desc.pStaticSamplers = &sampler;

    ID3DBlob *blob = nullptr, *errors = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors);
    if (errors != nullptr)
        errors->Release();
    if (SUCCEEDED(hr))
    {
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&_root));
        blob->Release();
    }
    if (FAILED(hr))
    {
        _error = "could not create the root signature";
        return false;
    }

    const struct
    {
        const void *code;
        size_t size;
        ID3D12PipelineState **pipeline;
        const char *name;
    } shaders[] = {
        {prepare_cs, sizeof(prepare_cs), &_prepare, "the prepare pass"},
        {rcas_cs, sizeof(rcas_cs), &_rcas, "the sharpening pass"},
        {downsample_cs, sizeof(downsample_cs), &_downsample, "the downsample pass"},
        {motionblur_cs, sizeof(motionblur_cs), &_motionblur, "the motion blur pass"},
    };
    for (const auto &shader : shaders)
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso = {};
        pso.pRootSignature = _root;
        pso.CS = {shader.code, shader.size};
        if (FAILED(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(shader.pipeline))))
        {
            _error = std::string("could not create ") + shader.name;
            return false;
        }
    }

    D3D12_DESCRIPTOR_HEAP_DESC heap = {};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = SLOT_SIZE * SLOT_COUNT;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&_heap))))
    {
        _error = "could not create a descriptor heap";
        return false;
    }
    _increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

void passes::shutdown()
{
    release_textures(UINT64_MAX);
    for (retired &r : _retired)
        r.resource->Release();
    _retired.clear();
    release(_heap);
    release(_motionblur);
    release(_downsample);
    release(_rcas);
    release(_prepare);
    release(_root);
    _device = nullptr;
}

ID3D12Resource *passes::make_texture(DXGI_FORMAT format, uint32_t width, uint32_t height, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource *texture = nullptr;
    _device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&texture));
    return texture;
}

void passes::release_textures(uint64_t frame)
{
    for (ID3D12Resource **t : {&_color, &_motion, &_depth, &_downsampled, &_output, &_sharpened, &_blurred, &_presented})
    {
        if (*t != nullptr)
            _retired.push_back({*t, frame});
        *t = nullptr;
    }
}

bool passes::resize(uint32_t width, uint32_t height, uint32_t output_width, uint32_t output_height,
                    DXGI_FORMAT color_format, uint64_t frame)
{
    if (_color != nullptr && width == _width && height == _height && output_width == _output_width &&
        output_height == _output_height && color_format == _color_format)
        return true;

    release_textures(frame);
    _width = width;
    _height = height;
    _output_width = output_width;
    _output_height = output_height;
    _color_format = color_format;

    const D3D12_RESOURCE_STATES uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    _color = make_texture(color_format, width, height, uav);
    _motion = make_texture(DXGI_FORMAT_R16G16_FLOAT, width, height, uav);
    _depth = make_texture(DXGI_FORMAT_R32_FLOAT, width, height, uav);
    _output = make_texture(color_format, output_width, output_height, uav);
    _sharpened = make_texture(color_format, output_width, output_height, uav);
    _blurred = make_texture(color_format, output_width, output_height, uav);
    bool ok = _color && _motion && _depth && _output && _sharpened && _blurred;
    if (upscaling())
    {
        _downsampled = make_texture(color_format, width, height, uav);
        _presented = make_texture(color_format, output_width, output_height, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        ok = ok && _downsampled && _presented;
    }
    if (!ok)
    {
        char text[96];
        snprintf(text, sizeof(text), "could not allocate work textures for %ux%u to %ux%u", width, height,
                 output_width, output_height);
        _error = text;
        release_textures(frame);
        _width = _height = _output_width = _output_height = 0;
        return false;
    }
    _error.clear();
    return true;
}

void passes::collect(uint64_t frame)
{
    for (size_t i = 0; i < _retired.size();)
    {
        if (frame >= _retired[i].frame + RETIRE_FRAMES)
        {
            _retired[i].resource->Release();
            _retired[i] = _retired.back();
            _retired.pop_back();
        }
        else
        {
            ++i;
        }
    }
}

D3D12_CPU_DESCRIPTOR_HANDLE passes::offset(D3D12_CPU_DESCRIPTOR_HANDLE base, uint32_t index) const
{
    base.ptr += SIZE_T(index) * _increment;
    return base;
}

D3D12_GPU_DESCRIPTOR_HANDLE passes::next_slot(D3D12_CPU_DESCRIPTOR_HANDLE &cpu)
{
    const uint32_t first = (_next_slot++ % SLOT_COUNT) * SLOT_SIZE;
    cpu = offset(_heap->GetCPUDescriptorHandleForHeapStart(), first);
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = _heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += UINT64(first) * _increment;
    return gpu;
}

// A null resource still needs a format and a dimension, and reads as zeros.
void passes::write_srv(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource, DXGI_FORMAT format)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC desc = {};
    desc.Format = format;
    desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.Texture2D.MipLevels = 1;
    _device->CreateShaderResourceView(resource, &desc, at);
}

void passes::write_uav(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource, DXGI_FORMAT format)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc = {};
    desc.Format = format;
    desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    _device->CreateUnorderedAccessView(resource, nullptr, &desc, at);
}

// One source in our color format and one target, as RCAS, the downsample pass
// and the delivery to the game's FSR 1 output all have.
void passes::dispatch(ID3D12GraphicsCommandList *list, ID3D12PipelineState *pipeline, const void *constants,
                      uint32_t constant_count, ID3D12Resource *source, ID3D12Resource *target,
                      DXGI_FORMAT target_format, uint32_t width, uint32_t height)
{
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
    const D3D12_GPU_DESCRIPTOR_HANDLE table = next_slot(cpu);
    write_srv(offset(cpu, 0), source, _color_format);
    write_srv(offset(cpu, 1), nullptr, _color_format);
    write_srv(offset(cpu, 2), nullptr, _color_format);
    write_uav(offset(cpu, 3), target, target_format);
    write_uav(offset(cpu, 4), nullptr, _color_format);
    write_uav(offset(cpu, 5), nullptr, _color_format);

    list->SetDescriptorHeaps(1, &_heap);
    list->SetComputeRootSignature(_root);
    list->SetPipelineState(pipeline);
    if (constant_count != 0)
        list->SetComputeRoot32BitConstants(0, constant_count, constants, 0);
    list->SetComputeRootDescriptorTable(1, table);
    list->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
}

void passes::blur(ID3D12GraphicsCommandList *list, ID3D12Resource *source, float strength, float max_length,
                  uint64_t frame)
{
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
    const D3D12_GPU_DESCRIPTOR_HANDLE table = next_slot(cpu);
    write_srv(offset(cpu, 0), source, _color_format);
    write_srv(offset(cpu, 1), _motion, DXGI_FORMAT_R16G16_FLOAT);
    write_srv(offset(cpu, 2), nullptr, _color_format);
    write_uav(offset(cpu, 3), _blurred, _color_format);
    write_uav(offset(cpu, 4), nullptr, _color_format);
    write_uav(offset(cpu, 5), nullptr, _color_format);

    const float constants[4] = {strength, max_length, 16.0f, float(frame % 64)};
    list->SetDescriptorHeaps(1, &_heap);
    list->SetComputeRootSignature(_root);
    list->SetPipelineState(_motionblur);
    list->SetComputeRoot32BitConstants(0, 4, constants, 0);
    list->SetComputeRootDescriptorTable(1, table);
    list->Dispatch((_output_width + 7) / 8, (_output_height + 7) / 8, 1);
}

void passes::prepare(ID3D12GraphicsCommandList *list, const game_texture &color, const game_texture &motion,
                     const game_texture &depth, const prepare_constants &constants)
{
    D3D12_CPU_DESCRIPTOR_HANDLE cpu;
    const D3D12_GPU_DESCRIPTOR_HANDLE table = next_slot(cpu);
    write_srv(offset(cpu, 0), color.resource, color.format);
    write_srv(offset(cpu, 1), motion.resource, motion.format);
    write_srv(offset(cpu, 2), depth.resource, depth.format);
    write_uav(offset(cpu, 3), _color, _color_format);
    write_uav(offset(cpu, 4), _motion, DXGI_FORMAT_R16G16_FLOAT);
    write_uav(offset(cpu, 5), _depth, DXGI_FORMAT_R32_FLOAT);

    list->SetDescriptorHeaps(1, &_heap);
    list->SetComputeRootSignature(_root);
    list->SetPipelineState(_prepare);
    list->SetComputeRoot32BitConstants(0, sizeof(constants) / 4, &constants, 0);
    list->SetComputeRootDescriptorTable(1, table);
    list->Dispatch((_width + 7) / 8, (_height + 7) / 8, 1);

    const D3D12_RESOURCE_BARRIER barriers[] = {
        transition(_color, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(_motion, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        transition(_depth, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
    };
    list->ResourceBarrier(UINT(std::size(barriers)), barriers);
}

void passes::finish(ID3D12GraphicsCommandList *list, float sharpness, bool hdr, ID3D12Resource *taa_target, float blur_strength,
                    float blur_limit, uint64_t frame)
{
    ID3D12Resource *result = _output;
    if (sharpness > 0.0f)
    {
        const D3D12_RESOURCE_BARRIER readable =
            transition(_output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &readable);
        const float constants[4] = {sharpness, hdr ? 1.0f : 0.0f, 0.0f, 0.0f};
        dispatch(list, _rcas, constants, 4, _output, _sharpened, _color_format, _output_width, _output_height);
        const D3D12_RESOURCE_BARRIER back =
            transition(_output, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1, &back);
        result = _sharpened;
    }

    // Motion blur goes after the sharpening, so RCAS doesn't sharpen the
    // streaks. _motion is still readable from prepare().
    if (blur_strength > 0.0f)
    {
        const D3D12_RESOURCE_BARRIER readable =
            transition(result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &readable);
        blur(list, result, blur_strength, blur_limit, frame);
        const D3D12_RESOURCE_BARRIER back =
            transition(result, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1, &back);
        result = _blurred;
    }

    const D3D12_RESOURCE_BARRIER to_read = transition(result, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, READABLE);
    list->ResourceBarrier(1, &to_read);

    // The render-size picture for the game's TAA target: DLSS's own at DLAA,
    // the upscaled one brought back down otherwise.
    ID3D12Resource *for_taa = result;
    if (upscaling())
    {
        dispatch(list, _downsample, nullptr, 0, result, _downsampled, _color_format, _width, _height);
        const D3D12_RESOURCE_BARRIER barriers[] = {
            transition(_downsampled, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(_presented, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
        };
        list->ResourceBarrier(UINT(std::size(barriers)), barriers);
        list->CopyResource(_presented, result);
        const D3D12_RESOURCE_BARRIER presented =
            transition(_presented, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &presented);
        for_taa = _downsampled;
    }

    // The game draws its TAA into this target, so it is a render target now
    // and has to be one again for whatever reads it next.
    const D3D12_RESOURCE_BARRIER to_copy =
        transition(taa_target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResourceBarrier(1, &to_copy);
    list->CopyResource(taa_target, for_taa);

    D3D12_RESOURCE_BARRIER restore[3] = {
        transition(taa_target, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_RENDER_TARGET),
        transition(result, READABLE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    UINT count = 2;
    if (upscaling())
        restore[count++] = transition(_downsampled, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(count, restore);
    abandon(list);
}

void passes::abandon(ID3D12GraphicsCommandList *list)
{
    const D3D12_RESOURCE_BARRIER writable[] = {
        transition(_color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        transition(_motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        transition(_depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    };
    list->ResourceBarrier(UINT(std::size(writable)), writable);
}

void passes::deliver(ID3D12GraphicsCommandList *list, ID3D12Resource *target, DXGI_FORMAT format, uint32_t width,
                     uint32_t height)
{
    // The downsample pass samples by position, so it fits any size, and at one
    // to one it is a plain copy that converts formats.
    dispatch(list, _downsample, nullptr, 0, _presented, target, format, width, height);
    D3D12_RESOURCE_BARRIER written = {};
    written.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    written.UAV.pResource = target;
    list->ResourceBarrier(1, &written);
}
