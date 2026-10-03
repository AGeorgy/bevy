import bevy_render::color_operations::{
    oklch_to_linear_rgb,
    hsv_to_linear_rgb,
    hsl_to_linear_rgb,
    oklab_to_linear_rgb,
    okhsl_to_linear_rgb,
    srgb_to_linear,
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
