import bevy_render::color_operations::{
    oklch_to_linear_rgb,
    hsv_to_linear_rgb,
    hsl_to_linear_rgb,
    oklab_to_linear_rgb,
    okhsl_to_linear_rgb,
    srgb_to_linear,
    mix_oklch,
    mix_oklch_long,
    mix_hsv,
    mix_hsv_long,
    mix_hsl,
    mix_hsl_long,
    mix_okhsl,
    mix_okhsl_long,
};

/// Convert a color from the interpolation color space to linear RGBA.
fn convert_to_linear_rgba(color: vec4<f32>) -> vec4<f32> {
@if(IN_OKLCH || IN_OKLCH_LONG)
    let rgb = oklch_to_linear_rgb(color.xyz);
@elif(IN_OKHSL || IN_OKHSL_LONG)
    let rgb = okhsl_to_linear_rgb(color.xyz);
@elif(IN_HSV || IN_HSV_LONG)
    let rgb = hsv_to_linear_rgb(color.xyz);
@elif(IN_HSL || IN_HSL_LONG)
    let rgb = hsl_to_linear_rgb(color.xyz);
@elif(IN_OKLAB)
    let rgb = oklab_to_linear_rgb(color.xyz);
@elif(IN_SRGB)
    let rgb = srgb_to_linear(color.xyz);
@else
    let rgb = color.xyz;
    return vec4(rgb, color.a);
}

/// Mix the colors, choosing the appropriate interpolation method for the given color space
fn mix_colors(
    start_color: vec3<f32>,
    end_color: vec3<f32>,
    t: f32,
) -> vec3<f32> {
@if(IN_OKLCH)
    return mix_oklch(start_color, end_color, t);
@elif(IN_OKLCH_LONG)
    return mix_oklch_long(start_color, end_color, t);
@elif(IN_HSV)
    return mix_hsv(start_color, end_color, t);
@elif(IN_HSV_LONG)
    return mix_hsv_long(start_color, end_color, t);
@elif(IN_HSL)
    return mix_hsl(start_color, end_color, t);
@elif(IN_HSL_LONG)
    return mix_hsl_long(start_color, end_color, t);
@elif(IN_OKHSL)
    return mix_okhsl(start_color, end_color, t);
@elif(IN_OKHSL_LONG)
    return mix_okhsl_long(start_color, end_color, t);
@else
    // Just lerp in linear RGBA, OkLab and SRGBA spaces
    return mix(start_color, end_color, t);
}
