//! Horizon WebAssembly browser client entrypoint.
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x3,
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [f32; 16],
    model: [f32; 16],
    light_dir: [f32; 4],
    time: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

struct GpuContext {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    cube_vbuf: wgpu::Buffer,
    cube_ibuf: wgpu::Buffer,
    cube_indices: u32,
    ground_vbuf: wgpu::Buffer,
    ground_ibuf: wgpu::Buffer,
    ground_indices: u32,
    cube_uniform_buf: wgpu::Buffer,
    cube_bind_group: wgpu::BindGroup,
    ground_uniform_buf: wgpu::Buffer,
    ground_bind_group: wgpu::BindGroup,
    start_time: f64,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    is_dragging: bool,
    last_mouse: Option<(f64, f64)>,
}

fn current_time_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: config.width.max(1),
        height: config.height.max(1),
        depth_or_array_layers: 1,
    };
    let desc = wgpu::TextureDescriptor {
        label: Some("Horizon Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };
    let texture = device.create_texture(&desc);
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

impl GpuContext {
    async fn new(window: Arc<Window>) -> Result<Self, String> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::all();
        let instance = wgpu::Instance::new(descriptor);

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("Surface creation failed: {e}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Adapter request failed: {e}"))?;

        let info = adapter.get_info();
        log::info!(
            "Graphics adapter chosen: {} (Backend: {:?}, Driver: {})",
            info.name,
            info.backend,
            info.driver
        );

        let limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Horizon Web Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Device request failed: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: if caps.present_modes.contains(&wgpu::PresentMode::AutoVsync) {
                wgpu::PresentMode::AutoVsync
            } else {
                caps.present_modes[0]
            },
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::default(),
        };
        surface.configure(&device, &config);

        let depth_view = create_depth_texture(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Horizon Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Horizon Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Horizon Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(Vertex::desc())],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let grass_green = [0.44, 0.73, 0.35];
        let dirt_brown = [0.45, 0.30, 0.18];
        let side_dirt = [0.55, 0.38, 0.24];

        let cube_vertices: &[Vertex] = &[
            // Top (+Y)
            Vertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 1.0, 0.0], color: grass_green },
            Vertex { position: [-0.5, 0.5,  0.5], normal: [0.0, 1.0, 0.0], color: grass_green },
            Vertex { position: [ 0.5, 0.5,  0.5], normal: [0.0, 1.0, 0.0], color: grass_green },
            Vertex { position: [ 0.5, 0.5, -0.5], normal: [0.0, 1.0, 0.0], color: grass_green },
            // Bottom (-Y)
            Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], color: dirt_brown },
            Vertex { position: [ 0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], color: dirt_brown },
            Vertex { position: [ 0.5, -0.5,  0.5], normal: [0.0, -1.0, 0.0], color: dirt_brown },
            Vertex { position: [-0.5, -0.5,  0.5], normal: [0.0, -1.0, 0.0], color: dirt_brown },
            // Front (+Z)
            Vertex { position: [-0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0], color: side_dirt },
            Vertex { position: [ 0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0], color: side_dirt },
            Vertex { position: [ 0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0], color: side_dirt },
            Vertex { position: [-0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0], color: side_dirt },
            // Back (-Z)
            Vertex { position: [ 0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], color: side_dirt },
            Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], color: side_dirt },
            Vertex { position: [-0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0], color: side_dirt },
            Vertex { position: [ 0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0], color: side_dirt },
            // Right (+X)
            Vertex { position: [ 0.5, -0.5,  0.5], normal: [1.0, 0.0, 0.0], color: side_dirt },
            Vertex { position: [ 0.5, -0.5, -0.5], normal: [1.0, 0.0, 0.0], color: side_dirt },
            Vertex { position: [ 0.5,  0.5, -0.5], normal: [1.0, 0.0, 0.0], color: side_dirt },
            Vertex { position: [ 0.5,  0.5,  0.5], normal: [1.0, 0.0, 0.0], color: side_dirt },
            // Left (-X)
            Vertex { position: [-0.5, -0.5, -0.5], normal: [-1.0, 0.0, 0.0], color: side_dirt },
            Vertex { position: [-0.5, -0.5,  0.5], normal: [-1.0, 0.0, 0.0], color: side_dirt },
            Vertex { position: [-0.5,  0.5,  0.5], normal: [-1.0, 0.0, 0.0], color: side_dirt },
            Vertex { position: [-0.5,  0.5, -0.5], normal: [-1.0, 0.0, 0.0], color: side_dirt },
        ];

        let cube_indices: &[u16] = &[
            0, 1, 2,  0, 2, 3,       // Top
            4, 5, 6,  4, 6, 7,       // Bottom
            8, 9, 10,  8, 10, 11,    // Front
            12, 13, 14,  12, 14, 15, // Back
            16, 17, 18,  16, 18, 19, // Right
            20, 21, 22,  20, 22, 23, // Left
        ];

        let cube_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(cube_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let cube_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Index Buffer"),
            contents: bytemuck::cast_slice(cube_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let ground_green = [0.30, 0.52, 0.22];
        let ground_vertices: &[Vertex] = &[
            Vertex { position: [-7.0, -1.4, -7.0], normal: [0.0, 1.0, 0.0], color: ground_green },
            Vertex { position: [-7.0, -1.4,  7.0], normal: [0.0, 1.0, 0.0], color: ground_green },
            Vertex { position: [ 7.0, -1.4,  7.0], normal: [0.0, 1.0, 0.0], color: ground_green },
            Vertex { position: [ 7.0, -1.4, -7.0], normal: [0.0, 1.0, 0.0], color: ground_green },
        ];

        let ground_indices: &[u16] = &[0, 1, 2, 0, 2, 3];

        let ground_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground Vertex Buffer"),
            contents: bytemuck::cast_slice(ground_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let ground_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground Index Buffer"),
            contents: bytemuck::cast_slice(ground_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let cube_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cube Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cube_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cube Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cube_uniform_buf.as_entire_binding(),
            }],
        });

        let ground_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ground Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ground_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ground Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ground_uniform_buf.as_entire_binding(),
            }],
        });

        // Notify web UI that GPU context is ready
        if let Some(_win) = web_sys::window() {
            let backend_name = format!("{:?}", info.backend);
            let escaped_name = info.name.replace('\'', "\\'");
            let code = format!(
                "if (window.horizonOnGpuReady) window.horizonOnGpuReady('{backend_name}', '{escaped_name}');"
            );
            let _ = js_sys::eval(&code);
        }

        Ok(Self {
            surface,
            device,
            queue,
            config,
            depth_view,
            pipeline,
            cube_vbuf,
            cube_ibuf,
            cube_indices: cube_indices.len() as u32,
            ground_vbuf,
            ground_ibuf,
            ground_indices: ground_indices.len() as u32,
            cube_uniform_buf,
            cube_bind_group,
            ground_uniform_buf,
            ground_bind_group,
            start_time: current_time_ms(),
            camera_yaw: 0.8,
            camera_pitch: 0.45,
            camera_distance: 4.2,
            is_dragging: false,
            last_mouse: None,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth_texture(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        let t = ((current_time_ms() - self.start_time) / 1000.0) as f32;

        let aspect = (self.config.width as f32) / (self.config.height as f32).max(1.0);
        let proj = glam::Mat4::perspective_rh(55.0_f32.to_radians(), aspect, 0.1, 100.0);

        let eye = glam::Vec3::new(
            self.camera_distance * self.camera_pitch.cos() * self.camera_yaw.sin(),
            self.camera_distance * self.camera_pitch.sin(),
            self.camera_distance * self.camera_pitch.cos() * self.camera_yaw.cos(),
        );
        let target = glam::Vec3::new(0.0, 0.0, 0.0);
        let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
        let view_proj = proj * view;

        // Block animation: smooth rotation and subtle float bob
        let bob = (t * 2.2).sin() * 0.06;
        let block_model = glam::Mat4::from_translation(glam::Vec3::new(0.0, bob, 0.0))
            * glam::Mat4::from_rotation_y(t * 0.7);

        let cube_uniforms = Uniforms {
            view_proj: view_proj.to_cols_array(),
            model: block_model.to_cols_array(),
            light_dir: [0.5, 1.0, 0.3, 0.0],
            time: t,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        self.queue.write_buffer(
            &self.cube_uniform_buf,
            0,
            bytemuck::cast_slice(&[cube_uniforms]),
        );

        let ground_model = glam::Mat4::IDENTITY;
        let ground_uniforms = Uniforms {
            view_proj: view_proj.to_cols_array(),
            model: ground_model.to_cols_array(),
            light_dir: [0.5, 1.0, 0.3, 0.0],
            time: t,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        self.queue.write_buffer(
            &self.ground_uniform_buf,
            0,
            bytemuck::cast_slice(&[ground_uniforms]),
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };

        let view_texture = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Minecraft Daylight Sky Blue (#78a7ff)
        let sky_clear = wgpu::Color {
            r: 0.470588,
            g: 0.654902,
            b: 1.0,
            a: 1.0,
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Horizon Sky & World Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_texture,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(sky_clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);

            // Draw spinning grass block
            pass.set_bind_group(0, &self.cube_bind_group, &[]);
            pass.set_vertex_buffer(0, self.cube_vbuf.slice(..));
            pass.set_index_buffer(self.cube_ibuf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.cube_indices, 0, 0..1);

            // Draw ground platform
            pass.set_bind_group(0, &self.ground_bind_group, &[]);
            pass.set_vertex_buffer(0, self.ground_vbuf.slice(..));
            pass.set_index_buffer(self.ground_ibuf.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.ground_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

#[derive(Default)]
struct HorizonWebApp {
    canvas_id: String,
    window: Option<Arc<Window>>,
    gpu: Rc<RefCell<Option<GpuContext>>>,
}

impl HorizonWebApp {
    fn new(canvas_id: &str) -> Self {
        Self {
            canvas_id: canvas_id.to_owned(),
            window: None,
            gpu: Rc::new(RefCell::new(None)),
        }
    }
}

impl ApplicationHandler for HorizonWebApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        use winit::platform::web::WindowAttributesExtWebSys;

        let window_target = web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.get_element_by_id(&self.canvas_id));

        let mut attributes = Window::default_attributes().with_title("Horizon Web");

        if let Some(canvas) = window_target.and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok()) {
            attributes = attributes.with_canvas(Some(canvas));
        } else {
            attributes = attributes.with_append(true);
        }

        match event_loop.create_window(attributes) {
            Ok(win) => {
                let win = Arc::new(win);
                self.window = Some(win.clone());
                log::info!("Horizon Web window created successfully.");

                let gpu_cell = self.gpu.clone();
                let win_clone = win.clone();

                wasm_bindgen_futures::spawn_local(async move {
                    log::info!("Requesting GPU device and surface configuration...");
                    match GpuContext::new(win_clone.clone()).await {
                        Ok(context) => {
                            log::info!("Horizon WGPU render engine active.");
                            *gpu_cell.borrow_mut() = Some(context);
                            win_clone.request_redraw();
                        }
                        Err(err) => {
                            log::error!("GPU initialization failed: {err}");
                            if let Some(_win) = web_sys::window() {
                                let _ = js_sys::eval(&format!(
                                    "if (window.horizonOnGpuError) window.horizonOnGpuError('{}');",
                                    err.replace('\'', "\\'")
                                ));
                            }
                        }
                    }
                });
            }
            Err(e) => {
                log::error!("Failed to create Web window: {e}");
            }
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                    gpu.render();
                }

                if let Some(win) = self.window.as_ref() {
                    win.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                        gpu.is_dragging = state == ElementState::Pressed;
                        if !gpu.is_dragging {
                            gpu.last_mouse = None;
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                    if gpu.is_dragging {
                        if let Some((last_x, last_y)) = gpu.last_mouse {
                            let dx = (position.x - last_x) as f32;
                            let dy = (position.y - last_y) as f32;
                            gpu.camera_yaw += dx * 0.008;
                            gpu.camera_pitch = (gpu.camera_pitch + dy * 0.008).clamp(-1.4, 1.4);
                        }
                        gpu.last_mouse = Some((position.x, position.y));
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 0.4,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.01,
                };
                if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                    gpu.camera_distance = (gpu.camera_distance - scroll_y).clamp(1.8, 12.0);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(KeyCode::KeyR) = event.physical_key {
                        if let Some(gpu) = self.gpu.borrow_mut().as_mut() {
                            gpu.camera_yaw = 0.8;
                            gpu.camera_pitch = 0.45;
                            gpu.camera_distance = 4.2;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Browser initialization entrypoint called from JavaScript module.
#[wasm_bindgen]
pub fn run_horizon(canvas_id: &str) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    log::info!("Horizon Minecraft WebAssembly engine initializing on canvas '#{}'...", canvas_id);

    let event_loop = EventLoop::new().map_err(|e| JsValue::from_str(&e.to_string()))?;

    use winit::platform::web::EventLoopExtWebSys;
    let app = HorizonWebApp::new(canvas_id);
    event_loop.spawn_app(app);

    Ok(())
}
