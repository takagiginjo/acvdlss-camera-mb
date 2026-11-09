#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace reshade::log
{
    enum class level;
}

struct ID3D12Device;

namespace reshade::api
{
    struct command_queue;
}

// Logs to acvdlss.log next to the game's exe; warnings and errors also to
// ReShade's log, where they get an "ACV DLSS: " in front.
void acv_log(reshade::log::level level, const char *message);
// The folder of the game's exe.
std::wstring game_folder();
// "310.9.16" from a file's version resource, or empty.
std::string file_version(const std::wstring &path);
// The first lines of the log: versions and the game's graphics settings. Also
// starts writing access violations to the log.
void log_start();
// At device creation: the GPU and driver, and the modules a report needs.
void log_adapter(ID3D12Device *device);
void log_modules();

struct settings
{
    std::atomic<bool> enabled{true};
    std::atomic<uint32_t> preset{0}; // NVIDIA's value: 0 lets DLSS choose, 1..13 are A..M
    std::atomic<float> sharpness{0.0f}; // 0..1, 0 skips RCAS
    // Only in ReShade.ini: detailed logging, every frame if need be.
    std::atomic<bool> debug{false};

    // --- Camera Motion Blur settings ---
    std::atomic<float> shutter{0.5f};
    std::atomic<float> max_length{0.15f};
    std::atomic<float> mode{0.0f};
};
extern settings g_settings;

enum class taa_state
{
    no_device,   // the game has not made a D3D12 device yet, or it is not D3D12
    unavailable, // DLSS or our own passes could not start; see message
    off,
    waiting,     // no TAA pass seen recently
    running,
    failed,      // DLSS stopped working this session; see message
};

struct taa_status
{
    taa_state state = taa_state::no_device;
    uint32_t width = 0, height = 0;               // what the game renders
    uint32_t output_width = 0, output_height = 0; // what DLSS makes of it
    uint32_t display_width = 0;
    bool hdr = false;
    bool camera_motion_only = false;
    std::string message;
};

void taa_register();
void taa_unregister();
// Once a frame, from present, with the size of the back buffer.
void taa_present(reshade::api::command_queue *queue, uint32_t display_width, uint32_t display_height);
// Throws DLSS's history away, as after it was switched back on: the game's
// own TAA ran in between.
void taa_restart();
taa_status taa_get_status();
// Set once DLSS has started: why nvngx_dlss.dll and nvngx_dlssnr.dll may not
// work together, or empty.
std::string dlss_versions_warning();
// Texture LOD bias, set on the game's samplers as they are created.
// Called as the add-on loads, which ReShade does inside D3D12CreateDevice,
// before the game has a device to build pipelines with. It may clear the
// pipeline cache.
void lod_register();
// Once a frame: marks a pipeline cache this session cleared as rebuilt after
// enough gameplay, which is frames where DLSS replaced the TAA.
void lod_frame(bool gameplay);
// The bias for how much smaller FSR renders inside the resolution scale's
// size, 1 with FSR off.
float lod_bias_for(float fsr_ratio);
// The bias the game's settings file asks for now, which a restart would bring.
float lod_bias_saved();
float lod_bias();             // 0 when none was applied
uint32_t lod_biased_samplers();
// With Debug on: the game's own biases on its tiling texture samplers.
std::string lod_game_biases();
