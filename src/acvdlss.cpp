// ACV DLSS
//
// Replaces Assassin's Creed Valhalla's TAA with NVIDIA DLSS. taa.cpp finds the
// pass and runs DLSS; this file is the add-on around it: settings, the toggle
// key and the ACV DLSS window in the ReShade overlay.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define ImTextureID ImU64
#include <imgui.h>
#include <reshade.hpp>

#include "acvdlss.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <iterator>
#include <string>

using namespace reshade::api;

extern "C" __declspec(dllexport) const char *NAME = "ACV DLSS";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Replaces Assassin's Creed Valhalla's anti-aliasing (TAA) with NVIDIA DLSS.";

static const char *SECTION = "ACVDLSS";

settings g_settings;

static effect_runtime *g_runtime = nullptr;
static bool g_overlay_open = false;

// NVIDIA's preset numbers are the index. Letters the current runtime no longer
// has still go through; DLSS treats them as Auto, and the label says so.
static const char *PRESETS[] = {
    "Auto",
    "A (removed, acts as Auto)",
    "B (removed, acts as Auto)",
    "C (removed, acts as Auto)",
    "D (removed, acts as Auto)",
    "E (CNN)",
    "F (CNN)",
    "G (reserved, acts as Auto)",
    "H (reserved, acts as Auto)",
    "I (reserved, acts as Auto)",
    "J (transformer, 1st gen)",
    "K (transformer, 1st gen)",
    "L (transformer, 2nd gen)",
    "M (transformer, 2nd gen)",
};

//----------------------------------------------------------------------------
// The toggle key: one key, picked with a button.

static int g_toggle_key = VK_F10;
static bool g_listening = false;
static int g_listen_frame = 0; // ImGui frame the key button was last drawn on

static void save_preset(uint32_t preset)
{
    const char letter[2] = {char('A' + preset - 1), '\0'};
    reshade::set_config_value(nullptr, SECTION, "Preset", preset == 0 ? "Auto" : letter);
}

static uint32_t read_preset()
{
    char value[16] = "";
    size_t size = sizeof(value);
    if (!reshade::get_config_value(nullptr, SECTION, "Preset", value, &size))
        return 0;
    if (value[0] >= '0' && value[0] <= '9')
    {
        const unsigned long number = strtoul(value, nullptr, 10);
        const uint32_t preset = number < std::size(PRESETS) ? uint32_t(number) : 0;
        save_preset(preset);
        return preset;
    }
    const char letter = char(toupper(static_cast<unsigned char>(value[0])));
    if (value[1] == '\0' && letter >= 'A' && letter < char('A' + std::size(PRESETS) - 1))
        return uint32_t(letter - 'A' + 1);
    return 0;
}

static void load_settings()
{
    bool enabled = true, debug = false;
    int sharpness = 0;
    float shutter = 0.5f;
    float max_length = 0.15f;
    int mode = 0;

    reshade::get_config_value(nullptr, SECTION, "Enabled", enabled);
    const uint32_t preset = read_preset();
    reshade::get_config_value(nullptr, SECTION, "Sharpness", sharpness);
    reshade::get_config_value(nullptr, SECTION, "Debug", debug);
    
    // Load new motion blur settings
    reshade::get_config_value(nullptr, SECTION, "Shutter", shutter);
    reshade::get_config_value(nullptr, SECTION, "MaxLength", max_length);
    reshade::get_config_value(nullptr, SECTION, "MBMode", mode);

    g_settings.enabled = enabled;
    g_settings.debug = debug;
    g_settings.preset = preset;
    g_settings.sharpness = float(std::clamp(sharpness, 0, 100)) / 100.0f;
    g_settings.shutter = shutter;
    g_settings.max_length = max_length;
    g_settings.mode = float(mode);

    int key = g_toggle_key;
    reshade::get_config_value(nullptr, SECTION, "ToggleKey", key);
    g_toggle_key = key >= 0 && key < 256 ? key : VK_F10;
}

static std::string key_name(uint32_t vk);

static void log_settings()
{
    char text[160];
    snprintf(text, sizeof(text), "settings: %s, preset %s, sharpness %d%%, toggle key %s%s",
             g_settings.enabled ? "on" : "off", PRESETS[g_settings.preset],
             int(std::lround(g_settings.sharpness * 100.0f)),
             g_toggle_key != 0 ? key_name(g_toggle_key).c_str() : "none", g_settings.debug ? ", Debug on" : "");
    acv_log(reshade::log::level::info, text);
}

static void save_toggle_key()
{
    reshade::set_config_value(nullptr, SECTION, "ToggleKey", g_toggle_key);
}

static const auto INDICATOR_TIME = std::chrono::seconds(3);
static std::chrono::steady_clock::time_point g_indicator_until;

static void log_change(const char *format, ...)
{
    char text[160];
    va_list args;
    va_start(args, format);
    vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    acv_log(reshade::log::level::info, text);
}

static void set_enabled(bool enabled, const char *how)
{
    log_change("DLSS switched %s with %s", enabled ? "on" : "off", how);
    g_settings.enabled = enabled;
    reshade::set_config_value(nullptr, SECTION, "Enabled", enabled);
    if (enabled)
        taa_restart();
    g_indicator_until = std::chrono::steady_clock::now() + INDICATOR_TIME;
}

static bool toggle_key_pressed()
{
    return g_toggle_key != 0 && g_runtime != nullptr && g_runtime->is_key_pressed(g_toggle_key);
}

static bool is_reshade_key(int vk)
{
    for (const char *name : {"KeyOverlay", "KeyEffects", "KeyScreenshot", "KeyReload", "KeyNextPreset",
                             "KeyPreviousPreset", "KeyFPS", "KeyFrametime"})
    {
        char value[64] = "";
        size_t size = sizeof(value);
        unsigned key = 0;
        if (reshade::get_config_value(nullptr, "INPUT", name, value, &size) && sscanf(value, "%u", &key) == 1 &&
            int(key) == vk)
            return true;
    }
    return false;
}

static std::string utf8(const wchar_t *text)
{
    char out[64];
    const int n = WideCharToMultiByte(CP_UTF8, 0, text, -1, out, sizeof(out), nullptr, nullptr);
    return n > 0 ? std::string(out) : std::string();
}

static std::string key_name(uint32_t vk)
{
    if ((vk >= '0' && vk <= '9') || (vk >= 'A' && vk <= 'Z'))
        return std::string(1, char(vk));
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9)
        return "Numpad " + std::to_string(vk - VK_NUMPAD0);
    if (vk >= VK_F1 && vk <= VK_F24)
        return "F" + std::to_string(vk - VK_F1 + 1);
    switch (vk)
    {
    case VK_MBUTTON: return "Middle Mouse";
    case VK_XBUTTON1: return "Mouse 4";
    case VK_XBUTTON2: return "Mouse 5";
    case VK_PAUSE: return "Pause";
    case VK_CAPITAL: return "Caps Lock";
    case VK_ESCAPE: return "Escape";
    case VK_SPACE: return "Space";
    case VK_RETURN: return "Enter";
    case VK_TAB: return "Tab";
    case VK_PRIOR: return "Page Up";
    case VK_NEXT: return "Page Down";
    case VK_END: return "End";
    case VK_HOME: return "Home";
    case VK_LEFT: return "Left";
    case VK_UP: return "Up";
    case VK_RIGHT: return "Right";
    case VK_DOWN: return "Down";
    case VK_SNAPSHOT: return "Print Screen";
    case VK_INSERT: return "Insert";
    case VK_DELETE: return "Delete";
    case VK_MULTIPLY: return "Numpad *";
    case VK_ADD: return "Numpad +";
    case VK_SUBTRACT: return "Numpad -";
    case VK_DECIMAL: return "Numpad .";
    case VK_DIVIDE: return "Numpad /";
    case VK_NUMLOCK: return "Num Lock";
    case VK_SCROLL: return "Scroll Lock";
    }
    if (const UINT scan = MapVirtualKeyW(vk, MAPVK_VK_TO_VSC); scan != 0)
    {
        wchar_t text[48] = L"";
        if (GetKeyNameTextW(LONG(scan << 16), text, int(std::size(text))) > 0)
            return utf8(text);
    }
    return "Key " + std::to_string(vk);
}

static void draw_toggle_key(effect_runtime *runtime)
{
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Toggle key");
    ImGui::SameLine();
    const std::string name = g_toggle_key != 0 ? key_name(g_toggle_key) : std::string("None");
    const std::string label = (g_listening ? std::string("Press a key, Esc to cancel") : name) + "###toggle_key";
    const int frame = ImGui::GetFrameCount();
    if (g_listening && frame != g_listen_frame + 1)
        g_listening = false;
    bool started = false;
    if (ImGui::Button(label.c_str(), ImVec2(220.0f, 0.0f)) && !g_listening)
        g_listening = started = true;
    g_listen_frame = frame;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Switches DLSS on and off while playing.\nBackspace while picking leaves it without a key.");

    if (!g_listening || started || runtime == nullptr)
        return;
    for (int vk = VK_MBUTTON; vk < 256; ++vk)
    {
        if (!runtime->is_key_pressed(vk) || vk == VK_SHIFT || vk == VK_CONTROL || vk == VK_MENU)
            continue;
        if (vk != VK_ESCAPE && is_reshade_key(vk))
            continue;
        if (vk != VK_ESCAPE)
        {
            g_toggle_key = vk == VK_BACK ? 0 : vk;
            save_toggle_key();
            log_change("toggle key set to %s", g_toggle_key != 0 ? key_name(g_toggle_key).c_str() : "none");
        }
        g_listening = false;
        break;
    }
}

static const ImVec4 WARNING_COLOR(0.96f, 0.73f, 0.24f, 1.0f);

static std::string describe_mode(const taa_status &s)
{
    char text[64];
    if (s.output_width == s.width && s.output_height == s.height)
        snprintf(text, sizeof(text), "DLAA %ux%u", s.width, s.height);
    else
        snprintf(text, sizeof(text), "%ux%u to %ux%u", s.width, s.height, s.output_width, s.output_height);
    return std::string(text) + (s.hdr ? ", HDR" : "");
}

static void draw_lod_bias()
{
    static float wanted = lod_bias();
    static auto read = std::chrono::steady_clock::time_point();
    const auto now = std::chrono::steady_clock::now();
    if (now - read >= std::chrono::seconds(1))
    {
        wanted = lod_bias_saved();
        read = now;
    }
    if (lod_bias() == 0.0f && wanted == 0.0f)
        return;
    ImGui::TextDisabled("LOD bias: %+.2f", lod_bias());
    if (std::abs(wanted - lod_bias()) > 0.01f)
    {
        ImGui::SameLine();
        ImGui::TextColored(WARNING_COLOR, "(%+.2f after a restart)", wanted);
    }
}

static void draw_status(const taa_status &s)
{
    switch (s.state)
    {
    case taa_state::no_device:
        ImGui::TextDisabled("Waiting for the game to start rendering.");
        break;
    case taa_state::unavailable:
        ImGui::TextColored(WARNING_COLOR, "DLSS is unavailable: %s.", s.message.c_str());
        break;
    case taa_state::off:
        ImGui::TextDisabled("Off. The game's own anti-aliasing is running.");
        break;
    case taa_state::waiting:
        ImGui::TextDisabled("Waiting for the game's anti-aliasing pass.");
        ImGui::TextDisabled("Anti-aliasing has to be on High in the game's graphics options.");
        break;
    case taa_state::running:
        ImGui::Text("Running: %s.", describe_mode(s).c_str());
        if (s.camera_motion_only)
            ImGui::TextDisabled("This scene has no motion vectors, so DLSS follows the camera alone.");
        break;
    case taa_state::failed:
