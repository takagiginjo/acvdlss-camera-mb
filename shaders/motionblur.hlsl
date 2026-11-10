// Camera and object motion blur, run on the finished DLSS image before the HUD
// is drawn. Each pixel is averaged along its own motion vector (the same
// vectors DLSS uses, in UV units, from prepare.hlsl), as if the shutter had
// stayed open for `strength` of a frame.

cbuffer constants : register(b0)
{
    float strength;   // shutter: the part of a frame's motion that is blurred, 0 to 1
    float max_length; // longest blur, as a fraction of the screen height
    float samples;    // taps along the blur
    float frame;      // changes every frame, moves the noise
};

Texture2D<float4> source : register(t0);
Texture2D<float2> motion : register(t1);
SamplerState linear_clamp : register(s0);
RWTexture2D<float4> target : register(u0);

// Interleaved gradient noise: spreads the taps so a few of them don't band.
float noise(uint2 p, float f)
{
    return frac(52.9829189 * frac(dot(float2(p) + f * 5.588238, float2(0.06711056, 0.00583715))));
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    target.GetDimensions(width, height);
    if (id.x >= width || id.y >= height)
        return;

    // The motion vectors are at the game's render size, the image at DLSS's
    // output size. Both cover the whole screen, so they share UVs.
    uint mw, mh;
    motion.GetDimensions(mw, mh);
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    int2 mp = int2(min(uint2(uv * float2(mw, mh)), uint2(mw - 1, mh - 1)));
    float2 v = motion.Load(int3(mp, 0)) * strength;

    float aspect = float(width) / float(height);
    float len = length(v * float2(aspect, 1.0));
    float limit = len > max_length ? max_length / len : 1.0;
    v *= limit;
    len *= limit;

    // Less than a pixel of motion: nothing to blur.
    float4 here = source[id.xy];
    if (len * float(height) < 1.0)
    {
        target[id.xy] = here;
        return;
    }

    int n = clamp(int(samples), 2, 32);
    float jitter = noise(id.xy, frame) - 0.5;
    float3 sum = float3(0.0, 0.0, 0.0);
    [loop]
    for (int i = 0; i < n; ++i)
    {
        // From half a blur behind to half ahead, each tap moved a little by
        // the noise inside its own slot.
        float t = (float(i) + 0.5 + jitter) / float(n) - 0.5;
        sum += source.SampleLevel(linear_clamp, uv + v * t, 0.0).rgb;
    }
    target[id.xy] = float4(sum / float(n), here.a);
}
