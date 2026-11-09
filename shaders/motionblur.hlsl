// Camera motion blur, run on the DLSS image (after RCAS) before the HUD is
// drawn. Streaks each pixel along its motion vector, centered on the pixel,
// with a fixed number of jittered taps.
//
// Mode 0 blurs only what moves because the camera moves: pixels whose own
// motion (game vector) disagrees with the camera's reprojection, such as the
// player, the horse or NPCs, are left sharp. Mode 1 uses the game's vectors as
// they are, so objects blur along their own motion too.
//
// Taps that sit clearly in front of the pixel are dropped, so a sharp
// foreground does not smear over the background behind it.

cbuffer constants : register(b0)
{
    float shutter;    // fraction of a frame's motion that is blurred, 0.5 is a 180 degree shutter
    float max_length; // longest streak, as a fraction of the screen height
    float mode;       // 0: camera motion only, 1: all motion
    float hdr;        // 1 when the image is scene referred
};

Texture2D<float4> source : register(t0);       // the frame, at output size
Texture2D<float2> game_motion : register(t1);  // render size, previous minus current position in UV
Texture2D<float4> aux : register(t2);          // render size: xy camera-only motion, z view depth
SamplerState linear_clamp : register(s0);
RWTexture2D<float4> target : register(u0);

static const int TAPS = 16;

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    target.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;

    uint mw, mh;
    game_motion.GetDimensions(mw, mh);
    const float2 out_size = float2(width, height);
    const float2 motion_size = float2(mw, mh);
    const int2 motion_last = int2(mw, mh) - 1;

    const float2 uv = (float2(id.xy) + 0.5) / out_size;
    const int2 mp = min(int2(uv * motion_size), motion_last);
    const float4 center = source[id.xy];

    const float2 game_mv = game_motion[mp];
    const float4 info = aux[mp];
    const float depth = info.z;

    float2 mv = game_mv;
    if (mode < 0.5)
    {
        // How much of this pixel's motion is the camera's. Static world gives
        // a ratio near 0; something that rides with the camera (the player)
        // has a game vector near 0 and a camera vector that is not, ratio 1.
        const float cam_px = length(info.xy * out_size);
        const float own_px = length((game_mv - info.xy) * out_size);
        const float ratio = own_px / max(cam_px, 1.0);
        mv = info.xy * (1.0 - smoothstep(0.2, 0.6, ratio));
    }

    float2 streak = mv * shutter * out_size; // in output pixels
    float length_px = length(streak);
    const float limit = max_length * float(height);
    if (length_px > limit)
    {
        streak *= limit / length_px;
        length_px = limit;
    }
    if (length_px < 1.0)
    {
        target[id.xy] = center;
        return;
    }
    const float2 streak_uv = streak / out_size;

    // Interleaved gradient noise moves the taps a little per pixel, which
    // trades banding for fine grain.
    const float noise = frac(52.9829189 * frac(dot(float2(id.xy), float2(0.06711056, 0.00583715))));

    // The pixel itself counts once.
    const float center_w = hdr > 0.0 ? 1.0 / (1.0 + max(center.r, max(center.g, center.b))) : 1.0;
    float3 sum = center.rgb * center_w;
    float total = center_w;

    [loop]
    for (int i = 0; i < TAPS; ++i)
    {
        const float t = (float(i) + noise) / float(TAPS) - 0.5;
        const float2 tap_uv = saturate(uv + streak_uv * t);
        const int2 tp = min(int2(tap_uv * motion_size), motion_last);

        // Fully kept unless the tap is clearly nearer than this pixel.
        const float nearer = (depth - aux[tp].z) / max(depth, 1e-3);
        const float keep = 1.0 - smoothstep(0.05, 0.2, nearer);

        const float3 c = source.SampleLevel(linear_clamp, tap_uv, 0.0).rgb;
        // Highlights get less weight when scene referred, as in RCAS, so one
        // bright pixel does not own the streak.
        const float w = keep * (hdr > 0.0 ? 1.0 / (1.0 + max(c.r, max(c.g, c.b))) : 1.0);
        sum += c * w;
        total += w;
    }

    target[id.xy] = float4(sum / total, center.a);
}
