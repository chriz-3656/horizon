struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    light_dir: vec4<f32>,
    time: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos4 = uniforms.model * vec4<f32>(in.position, 1.0);
    out.world_pos = world_pos4.xyz;
    let normal4 = uniforms.model * vec4<f32>(in.normal, 0.0);
    out.normal = normalize(normal4.xyz);
    out.color = in.color;
    out.clip_position = uniforms.view_proj * world_pos4;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var base_color = in.color;
    let n = abs(in.normal);

    // Vanilla Minecraft directional face brightness:
    // Top (+Y): 1.0, Bottom (-Y): 0.5, Z-sides: 0.8, X-sides: 0.6
    var brightness = 0.7;
    if in.normal.y > 0.5 {
        brightness = 1.0;
    } else if in.normal.y < -0.5 {
        brightness = 0.5;
    } else if n.z > 0.5 {
        brightness = 0.8;
    } else if n.x > 0.5 {
        brightness = 0.6;
    }

    // Grass block rim effect for sides: green border near top
    if in.normal.y < 0.5 && in.normal.y > -0.5 {
        let local_y = in.world_pos.y - uniforms.model[3].y;
        if local_y > 0.32 {
            base_color = vec3<f32>(0.44, 0.73, 0.35); // Grass green rim
        }
    }

    let final_color = base_color * brightness;
    return vec4<f32>(final_color, 1.0);
}
